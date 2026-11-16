// pinsim: replay a cloud OLAP request log with pin/unpin, as the LARA simulator
// (cloud-olap-cache-experiments, tests/measure_replacement_algorithm_hit_ratio.cpp) does.
//
// Input CSV: "request_type,obj_id,size" with a header; request_type 0 = access
// (the object is pinned until its matching unpin), 1 = unpin.
//   access: hit -> pin count + 1, miss -> evict unpinned objects, insert pinned
//   unpin : pin count - 1
//   objects larger than uncacheable x cache size are always misses and never cached
//           (their unpins are ignored), as in LARA (uncacheable = 0.9)
// Only accesses are requests. The policy is GDSF-GCA with pin=1.
//
// With --no-pin, unpins are skipped, pin is off and nothing is uncacheable except
// objects larger than the cache: the run must then match cachesim on the lcs.
//
// usage: pinsim <trace.csv> <cache size: bytes, or a fraction of the working set>
//               [-e gdsf-gca params] [--uncacheable 0.9] [--no-pin] [--name NAME]
//               [--hits-out FILE]   one byte (1 = hit, 0 = miss) per access that reached the
//                                   cache, in order; byte i belongs to cache request i + 1
//                                   (the n_req of the event log)
// output: one CSV line (header with --header)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "libCacheSim/cache.h"
#include "libCacheSim/evictionAlgo.h"
#include "libCacheSim/request.h"

struct Event {
  uint8_t type;
  uint64_t id;
  int64_t size;
};

static const double ALPHA = 46.73, BETA = 12.30;  // ms, ms/MiB (LARA eq. 1)

static double latency(int64_t size) { return ALPHA + BETA * (double)size / 1048576.0; }

int main(int argc, char **argv) {
  const char *path = nullptr, *size_arg = nullptr;
  std::string params, name, hits_out;
  double uncacheable = 0.9;
  bool pin = true, header = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-e" && i + 1 < argc) params = argv[++i];
    else if (a == "--uncacheable" && i + 1 < argc) uncacheable = atof(argv[++i]);
    else if (a == "--no-pin") pin = false;
    else if (a == "--name" && i + 1 < argc) name = argv[++i];
    else if (a == "--header") header = true;
    else if (a == "--hits-out" && i + 1 < argc) hits_out = argv[++i];
    else if (!path) path = argv[i];
    else if (!size_arg) size_arg = argv[i];
    else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
  }
  if (!path || !size_arg) {
    fprintf(stderr, "usage: %s <trace.csv> <cache size> [-e params] [--uncacheable 0.9] "
                    "[--no-pin] [--name NAME] [--header]\n", argv[0]);
    return 1;
  }
  if (name.empty()) name = path;

  // read the log; the working set is the sum of the first size of each accessed object
  FILE *f = fopen(path, "r");
  if (!f) { perror(path); return 1; }
  char line[256];
  if (!fgets(line, sizeof line, f) || strncmp(line, "request_type", 12) != 0) {
    fprintf(stderr, "%s: expected header request_type,obj_id,size\n", path);
    return 1;
  }
  std::vector<Event> log;
  std::unordered_set<uint64_t> seen;
  int64_t working_set = 0;
  while (fgets(line, sizeof line, f)) {
    unsigned type;
    unsigned long long id;
    long long size;
    if (sscanf(line, "%u,%llu,%lld", &type, &id, &size) != 3 || type > 1 || size <= 0) {
      fprintf(stderr, "%s: bad line: %s", path, line);
      return 1;
    }
    log.push_back({(uint8_t)type, id, size});
    if (type == 0 && seen.insert(id).second) working_set += size;
  }
  fclose(f);

  double s = atof(size_arg);
  int64_t cache_size = s <= 1.0 ? (int64_t)(s * (double)working_set) : (int64_t)s;
  if (!pin) uncacheable = 1.0;
  double limit = uncacheable * (double)cache_size;

  common_cache_params_t cp = default_common_cache_params();
  cp.cache_size = cache_size;
  std::string full = params;
  if (pin) full = full.empty() ? "pin=1" : "pin=1," + full;
  cache_t *cache = GDSF_GCA_init(cp, full.empty() ? nullptr : full.c_str());

  FILE *hits_f = hits_out.empty() ? nullptr : fopen(hits_out.c_str(), "wb");
  if (!hits_out.empty() && !hits_f) { perror(hits_out.c_str()); return 1; }
  request_t *req = new_request();
  int64_t n_req = 0, n_miss = 0, req_bytes = 0, miss_bytes = 0, n_uncacheable = 0;
  double lat_total = 0, lat_miss = 0;
  for (const Event &e : log) {
    bool big = (double)e.size > limit;
    if (e.type == 1) {
      if (pin && !big) GDSF_GCA_unpin(cache, e.id);
      continue;
    }
    n_req++;
    req_bytes += e.size;
    lat_total += latency(e.size);
    bool hit = false;
    if (big) {
      n_uncacheable++;
    } else {
      req->obj_id = e.id;
      req->obj_size = e.size;
      req->clock_time = n_req;
      req->valid = true;
      hit = cache->get(cache, req);
      if (hits_f) fputc(hit ? 1 : 0, hits_f);
    }
    if (!hit) {
      n_miss++;
      miss_bytes += e.size;
      lat_miss += latency(e.size);
    }
  }

  if (header)
    printf("trace,params,pin,cache_size,n_req,miss_ratio,byte_miss_ratio,uhr,latency_miss,"
           "n_uncacheable,n_evict_blocked\n");
  printf("%s,\"%s\",%d,%lld,%lld,%.6f,%.6f,%.6f,%.6f,%lld,%lld\n", name.c_str(), params.c_str(),
         pin, (long long)cache_size, (long long)n_req, (double)n_miss / n_req,
         (double)miss_bytes / req_bytes, 1 - lat_miss / lat_total, lat_miss / lat_total,
         (long long)n_uncacheable, (long long)GDSF_GCA_n_evict_blocked(cache));
  if (hits_f) fclose(hits_f);
  free_request(req);
  cache->cache_free(cache);
  return 0;
}
