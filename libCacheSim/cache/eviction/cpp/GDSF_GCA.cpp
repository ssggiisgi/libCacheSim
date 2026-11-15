/* GDSF_GCA: greedy dual frequency size */

#include <cassert>
#include <list>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include "abstractRank.hpp"

namespace eviction {

//ghost entry
struct ghost_entry {
  double slack_norm;
  int64_t bytes_admitted_at_evict;
  int64_t obj_size;
  int64_t fr_evict;  //축출될 때의 freq (resurrection용, 식 14)
};

class GDSF_GCA : public abstractRank {
 public:
  GDSF_GCA() = default;

  double clock = 0.0;          // clock base + rho(max - clock base)
  double rho = 1.0;            // rho 초기 값

  double clock_base = 0.0;
  double max_evicted_key = 0.0;
  int64_t n_evicted_in_epi = 0;

  //나중에 초기값 설정하기!!
  double log_rho = 0.0;
  int64_t bytes_admitted = 0; //전역 시간 역할
  bool learn=false; //false이면 rho 고정 (테스트용)
  //eta : 학습률
  //V(o) = freq(o) * cost(o) / size(o)^gamma
  //  gamma : size 지수 (1: size로 나눔, 0: size 무시)
  //  cost  : uniform이면 c0 = 1e6, latency이면 L(o) = lat_alpha + lat_beta * size(MiB) [ms]
  //  -> 초안의 V_OHR(식 3) = uniform, gamma=1 / V_BHR(식 4) = uniform, gamma=0
  //     V_Latency(식 5) = latency, gamma=1
  double eta = 0.05, w_mem = 1.0, w_burst = 0.25, w_res = 1.0;
  double gamma = 1.0;
  bool latency_cost = false;
  //cloud download latency 모델 (LARA 논문 식 1, cloud-olap-cache-experiments의 GDSF-Latency와 같은 상수)
  double lat_alpha = 46.73;  //ms, 요청당 고정 지연
  double lat_beta = 12.30;   //ms/MiB, 전송 지연
  double rho_min = 1e-6, rho_max = 4.0, ghost_ratio = 1.0;

  //resurrection (식 14): ghost hit으로 다시 들어오는 object의 freq를 감쇠해서 복원
  //  freq0 = round(1 + exp(-turns / w_res) * fr_evict), 기본값 끔, learn과 독립
  bool resurrect = false;
  int64_t pending_freq0 = 1;  //get()의 ghost hit에서 계산, 바로 다음 insert()에서 사용 후 1로 되돌림



  //ghost list
  std::list<obj_id_t> ghost_lru; //front에 넣고 back에서 축출
  std::unordered_map<obj_id_t, std::pair<ghost_entry, std::list<obj_id_t>::iterator>> ghost_map;
  int64_t ghost_bytes = 0;
  int64_t ghost_capacity = 0; // cache_size*ghost_ratio

  //rho log (rho-log=<path prefix>): cache 크기마다 csv 하나, rho-log-every 요청마다 한 줄
  std::string rho_log_prefix;
  int64_t rho_log_every = 100000;
  FILE *rho_log = nullptr;
  //구간 통계: 요청, miss, ghost hit, rho를 올린/내린 횟수
  int64_t win_req = 0, win_miss = 0, win_ghost_hit = 0, win_up = 0, win_down = 0;
  //구간 신호 크기의 합 (eta 곱하기 전, w*g): 올림은 w_burst 감쇠 후/전, 내림은 절댓값
  double win_up_sum = 0, win_up_sum_raw = 0, win_down_sum = 0;

  //event log (event-log=<path prefix>): 사건마다 한 줄, 단위 테스트용 (구현 검증용)
  std::string event_log_prefix;
  FILE *event_log = nullptr;
  int64_t cur_req = 0;  //지금 처리 중인 요청 번호 (ghost에서 밀려날 때 기록용)
  };
}
////////////////////////////////////// namespace eviction


#ifdef __cplusplus
extern "C" {
#endif

// ***********************************************************************
// ****                                                               ****
// ****                   function declarations                       ****
// ****                                                               ****
// ***********************************************************************

cache_t *GDSF_GCA_init(const common_cache_params_t ccache_params,
                   const char *params);
static void GDSF_GCA_free(cache_t *cache);
static bool GDSF_GCA_get(cache_t *cache, const request_t *req);

static cache_obj_t *GDSF_GCA_find(cache_t *cache, const request_t *req,
                              bool update_cache);
static cache_obj_t *GDSF_GCA_insert(cache_t *cache, const request_t *req);
static cache_obj_t *GDSF_GCA_to_evict(cache_t *cache, const request_t *req);
static void GDSF_GCA_evict(cache_t *cache, const request_t *req);

static bool GDSF_GCA_remove(cache_t *cache, obj_id_t obj_id);

// ***********************************************************************
// ****                                                               ****
// ****                   end user facing functions                   ****
// ****                                                               ****
// ****                       init, free, get                         ****
// ***********************************************************************

///////////////////////////////////////////////////
//ｐａｒａｍｅｔｅｒ ｐａｒｓｉｎｇ//
///////////////////////////////////////////////////
static void GDSF_GCA_parse_params(cache_t *cache, const char *params) {
  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
  //ｉｎｉｔｉａｌ ｖａｌｕｅ
  if (params == nullptr || params[0] == '\0') return;

  char *params_str = strdup(params);
  char *to_free = params_str;

  while (params_str != nullptr && params_str[0] != '\0') {
    char *key = strsep(&params_str, "=");
    char *value = strsep(&params_str, ",");
    while (params_str != nullptr && *params_str == ' ') params_str++;
    if (value == nullptr || value[0] == '\0')
      ERROR("GDSF_GCA: missing value for %s\n", key);

    if      (strcasecmp(key, "rho") == 0)         gdsf->rho = strtod(value, nullptr);
    else if (strcasecmp(key, "learn") == 0)       gdsf->learn = (atoi(value) != 0);
    else if (strcasecmp(key, "eta") == 0)         gdsf->eta = strtod(value, nullptr);
    else if (strcasecmp(key, "w-mem") == 0)       gdsf->w_mem = strtod(value, nullptr);
    else if (strcasecmp(key, "w-burst") == 0)     gdsf->w_burst = strtod(value, nullptr);
    else if (strcasecmp(key, "w-res") == 0)       gdsf->w_res = strtod(value, nullptr);
    else if (strcasecmp(key, "resurrect") == 0)   gdsf->resurrect = (atoi(value) != 0);
    else if (strcasecmp(key, "rho-min") == 0)     gdsf->rho_min = strtod(value, nullptr);
    else if (strcasecmp(key, "rho-max") == 0)     gdsf->rho_max = strtod(value, nullptr);
    else if (strcasecmp(key, "ghost-ratio") == 0) gdsf->ghost_ratio = strtod(value, nullptr);
    else if (strcasecmp(key, "gamma") == 0)       gdsf->gamma = strtod(value, nullptr);
    else if (strcasecmp(key, "cost") == 0) {
      if      (strcasecmp(value, "uniform") == 0) gdsf->latency_cost = false;
      else if (strcasecmp(value, "latency") == 0) gdsf->latency_cost = true;
      else { ERROR("GDSF_GCA: cost must be uniform or latency, got %s\n", value); }
    }
    else if (strcasecmp(key, "lat-alpha") == 0)   gdsf->lat_alpha = strtod(value, nullptr);
    else if (strcasecmp(key, "lat-beta") == 0)    gdsf->lat_beta = strtod(value, nullptr);
    else if (strcasecmp(key, "beta") == 0) {
      ERROR("GDSF_GCA: beta (size 지수) was renamed to gamma\n");
    }
    else if (strcasecmp(key, "rho-log") == 0)     gdsf->rho_log_prefix = value;
    else if (strcasecmp(key, "rho-log-every") == 0) gdsf->rho_log_every = atoll(value);
    else if (strcasecmp(key, "event-log") == 0)   gdsf->event_log_prefix = value;
    else ERROR("GDSF_GCA does not have parameter %s\n", key);
    }
  free(to_free);

  // 범위 검사는 모든 키를 다 읽은 뒤에
  if (gdsf->learn) {
    if (gdsf->rho_min <= 0 || gdsf->rho_min > gdsf->rho_max)
      ERROR("GDSF_GCA: need 0 < rho-min <= rho-max\n");
    if (gdsf->rho < gdsf->rho_min || gdsf->rho > gdsf->rho_max)
      ERROR("GDSF_GCA: with learn=1, rho must be in [rho-min, rho-max]\n");
  } else if (gdsf->rho < 0) {
    ERROR("GDSF_GCA: rho must be >= 0\n");
  }
  if (gdsf->eta < 0 || gdsf->w_mem <= 0 || gdsf->w_burst <= 0 || gdsf->ghost_ratio < 0)
    ERROR("GDSF_GCA: invalid eta / w-mem / w-burst / ghost-ratio\n");
  if (gdsf->w_res <= 0)
    ERROR("GDSF_GCA: w-res must be > 0\n");
  if (gdsf->gamma < 0)
    ERROR("GDSF_GCA: gamma must be >= 0\n");
  if (gdsf->lat_alpha < 0 || gdsf->lat_beta < 0 || gdsf->lat_alpha + gdsf->lat_beta <= 0)
    ERROR("GDSF_GCA: need lat-alpha >= 0, lat-beta >= 0, not both 0\n");
  if (gdsf->rho_log_every <= 0)
    ERROR("GDSF_GCA: rho-log-every must be > 0\n");
}
/////////////////////////////////////////////////

//////////////////////////////
//ghost erase
//////////////////////////////
static void ghost_erase(eviction::GDSF_GCA *g, decltype(g->ghost_map)::iterator it){
  g->ghost_bytes -= it->second.first.obj_size;
  g->ghost_lru.erase(it->second.second);
  g->ghost_map.erase(it);
}

//////////////////////////////
//event log: 한 사건을 한 줄로 기록 (해당 없는 칸은 비움)
//  n_req,event,obj_id,obj_size,freq,key,clock_base,value,slack,turns,s,g_raw,w,g,dlog_rho,rho,clock,clamped
//////////////////////////////
#define EV_NA NAN
static void event_log_write(eviction::GDSF_GCA *g, const char *event, int64_t obj_id,
                            int64_t obj_size, int64_t freq, double key, double clock_base,
                            double value, double slack, double turns, double s, double g_raw,
                            double w, double g_final, double dlog_rho, double rho, double clock,
                            int clamped) {
  if (g->event_log == nullptr) return;
  auto f = [&](double x) {
    if (std::isnan(x)) fputs(",", g->event_log);
    else fprintf(g->event_log, ",%.17g", x);
  };
  fprintf(g->event_log, "%lld,%s", (long long)g->cur_req, event);
  if (obj_id >= 0) fprintf(g->event_log, ",%lld", (long long)obj_id); else fputs(",", g->event_log);
  if (obj_size >= 0) fprintf(g->event_log, ",%lld", (long long)obj_size); else fputs(",", g->event_log);
  if (freq >= 0) fprintf(g->event_log, ",%lld", (long long)freq); else fputs(",", g->event_log);
  f(key); f(clock_base); f(value); f(slack); f(turns); f(s); f(g_raw); f(w); f(g_final);
  f(dlog_rho); f(rho); f(clock);
  if (clamped >= 0) fprintf(g->event_log, ",%d\n", clamped); else fputs(",\n", g->event_log);
}

//////////////////////////////
//ghost insert
//////////////////////////////
static void ghost_insert(eviction::GDSF_GCA *g, obj_id_t id, const eviction::ghost_entry &e){
  //이미 ghost에 있으면 먼저 제거. cachesim에서는 일어나지 않는다: 캐시에 들어오는 길(get의 miss)이
  //삽입 전에 ghost entry를 지우므로 캐시에 있는 id는 ghost에 없다. insert를 직접 부르는 경우 대비
  auto old = g->ghost_map.find(id);
  if (old != g->ghost_map.end()) ghost_erase(g, old);

  g->ghost_lru.push_front(id);
  g->ghost_map[id] = {e, g->ghost_lru.begin()};
  g->ghost_bytes += e.obj_size;
  //ghost list 넘치면 축출
  while (g->ghost_bytes > g->ghost_capacity && !g->ghost_lru.empty()){
    auto victim = g->ghost_map.find(g->ghost_lru.back());
    event_log_write(g, "ghost_drop", victim->first, victim->second.first.obj_size, -1,
                    EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA,
                    EV_NA, EV_NA, -1);
    ghost_erase(g, victim);
  }
}

////////////////////////////////
//gca_value V(o) 
////////////////////////////////
static inline double gca_value(const eviction::GDSF_GCA *g, int64_t freq, int64_t size){
  //cost(o): uniform이면 c0 = 1e6, latency이면 L(o) [ms]
  double cost = g->latency_cost
                    ? g->lat_alpha + g->lat_beta * ((double)size / 1024 / 1024)
                    : 1.0e6;
  if (g->gamma == 1.0) return (double)freq * cost / size;  //OHR, Latency
  if (g->gamma == 0.0) return (double)freq * cost;         //BHR
  return (double)freq * cost / pow((double)size, g->gamma);
}
//////////////////////////////
//rho log: 구간 통계 한 줄 쓰고 초기화
//////////////////////////////
static void rho_log_flush(eviction::GDSF_GCA *g, int64_t n_req){
  if (g->rho_log == nullptr || g->win_req == 0) return;
  double rho = g->learn ? exp(g->log_rho) : g->rho;
  fprintf(g->rho_log, "%lld,%.6g,%.6f,%lld,%lld,%lld,%.6g,%.6g,%.6g\n", (long long)n_req, rho,
          (double)g->win_miss / g->win_req, (long long)g->win_ghost_hit,
          (long long)g->win_up, (long long)g->win_down,
          g->win_up_sum, g->win_up_sum_raw, g->win_down_sum);
  g->win_req = g->win_miss = g->win_ghost_hit = g->win_up = g->win_down = 0;
  g->win_up_sum = g->win_up_sum_raw = g->win_down_sum = 0;
}
//////////////////////////////////////////////////

/**
 * @brief initialize the cache
 *
 * @param ccache_params some common cache parameters
 * @param params cache specific parameters, see parse_params
 * function or use -e "print" with the cachesim binary
 */
cache_t *GDSF_GCA_init(const common_cache_params_t ccache_params,
                   const char *params) {
  cache_t *cache = cache_struct_init("GDSF_GCA", ccache_params, params);
  cache->eviction_params = reinterpret_cast<void *>(new eviction::GDSF_GCA);
  GDSF_GCA_parse_params(cache, params);

  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);

  gdsf->log_rho = log(gdsf->rho);
  gdsf->ghost_capacity = (int64_t)(ccache_params.cache_size * gdsf->ghost_ratio);
  if (!gdsf->rho_log_prefix.empty()) {
    std::string path = gdsf->rho_log_prefix + "_" + std::to_string(ccache_params.cache_size) + ".csv";
    gdsf->rho_log = fopen(path.c_str(), "w");
    if (gdsf->rho_log == nullptr) ERROR("GDSF_GCA: cannot open rho log %s\n", path.c_str());
    fprintf(gdsf->rho_log, "n_req,rho,miss_ratio,ghost_hits,rho_up,rho_down,up_sum,up_sum_raw,down_sum\n");
  }
  if (!gdsf->event_log_prefix.empty()) {
    std::string path = gdsf->event_log_prefix + "_" + std::to_string(ccache_params.cache_size) + ".csv";
    gdsf->event_log = fopen(path.c_str(), "w");
    if (gdsf->event_log == nullptr) ERROR("GDSF_GCA: cannot open event log %s\n", path.c_str());
    fprintf(gdsf->event_log, "n_req,event,obj_id,obj_size,freq,key,clock_base,value,slack,"
            "turns,s,g_raw,w,g,dlog_rho,rho,clock,clamped\n");
  }
  cache->cache_init = GDSF_GCA_init;
  cache->cache_free = GDSF_GCA_free;
  cache->get = GDSF_GCA_get;
  cache->find = GDSF_GCA_find;
  cache->insert = GDSF_GCA_insert;
  cache->evict = GDSF_GCA_evict;
  cache->to_evict = GDSF_GCA_to_evict;
  cache->remove = GDSF_GCA_remove;

  if (ccache_params.consider_obj_metadata) {
    // freq + priority
    cache->obj_md_size = 8;
  } else {
    cache->obj_md_size = 0;
  }

  return cache;
}


/**
 * free resources used by this cache
 *
 * @param cache
 */
static void GDSF_GCA_free(cache_t *cache) {
  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
  if (gdsf->rho_log != nullptr) {
    rho_log_flush(gdsf, cache->n_req);  //마지막 구간
    fclose(gdsf->rho_log);
  }
  if (gdsf->event_log != nullptr) fclose(gdsf->event_log);
  delete gdsf;
  cache_struct_free(cache);
}

/**
 * @brief this function is the user facing API
 * it performs the following logic
 *
 * ```
 * if obj in cache:
 *    update_metadata
 *    return true
 * else:
 *    if cache does not have enough space:
 *        evict until it has space to insert
 *    insert the object
 *    return false
 * ```
 *
 * @param cache
 * @param req
 * @return true if cache hit, false if cache miss
 */
static bool GDSF_GCA_get(cache_t *cache, const request_t *req) {
  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
  cache_obj_t *obj = cache->find(cache, req, true);
  bool hit = (obj != NULL);
  gdsf->cur_req = cache->n_req;
  gdsf->win_req++;
  if (!hit) gdsf->win_miss++;

  //cache miss
  if (!hit && cache->can_insert(cache, req)) {
    //ghost look up
    auto it = gdsf->ghost_map.find(req->obj_id);
     //ghost에서 발견됨 -> rho 업데이트
    if (it != gdsf->ghost_map.end()){
      const eviction::ghost_entry &e = it->second.first;
      gdsf->win_ghost_hit++;
      double turns = (double)(gdsf->bytes_admitted - e.bytes_admitted_at_evict)/ cache->cache_size;
      if (gdsf->learn){
        double s = std::min(1.0, std::max(0.0, e.slack_norm));
        double g = 2.0 * s -1.0; //s가 0이면 -1(aging 늦추기), s가 1이면 1(aging 강하게)
        double g_raw = g;
        double w = exp(-turns / gdsf->w_mem); //금방 재요청된건지 아닌지 표현
        if (g > 0) gdsf->win_up_sum_raw += w * g;
        if (g>0) g *= exp(-turns / gdsf->w_burst); //g>0일때 한참 뒤에 재요청된 item의 영향 작게
        if (g > 0) { gdsf->win_up++; gdsf->win_up_sum += w * g; }
        else if (g < 0) { gdsf->win_down++; gdsf->win_down_sum -= w * g; }
        double log_rho_before = gdsf->log_rho;
        gdsf->log_rho += gdsf->eta *w *g;
        double log_rho_unclamped = gdsf->log_rho;
        gdsf->log_rho = std::clamp(gdsf->log_rho, log(gdsf->rho_min), log(gdsf->rho_max));
        event_log_write(gdsf, "ghost_hit", req->obj_id, e.obj_size, -1, EV_NA, EV_NA, EV_NA,
                        e.slack_norm, turns, s, g_raw, w, g, gdsf->log_rho - log_rho_before,
                        exp(gdsf->log_rho), EV_NA, gdsf->log_rho != log_rho_unclamped);
      } else {
        event_log_write(gdsf, "ghost_hit", req->obj_id, e.obj_size, -1, EV_NA, EV_NA, EV_NA,
                        e.slack_norm, turns, EV_NA, EV_NA, EV_NA, EV_NA, 0.0, gdsf->rho, EV_NA, 0);
      }
      //resurrection: 금방 돌아올수록 축출 당시 freq를 많이 돌려줌 (식 14, 반올림)
      if (gdsf->resurrect)
        gdsf->pending_freq0 = std::max<int64_t>(
            1, llround(1.0 + exp(-turns / gdsf->w_res) * (double)e.fr_evict));
      ghost_erase(gdsf,it);
    }
    cache->insert(cache, req);

    //episode begin
    gdsf->clock_base = gdsf->clock;
    gdsf->n_evicted_in_epi = 0;
    while (cache->get_occupied_byte(cache) > cache->cache_size) {
      cache->evict(cache, req);
    }
    //episode end

    //cache에서 evict X -> rho, clock 변화X
    //         evict O -> rho, clock 변화O
    if (gdsf->n_evicted_in_epi > 0){
      double rho = gdsf->learn ? exp(gdsf->log_rho) : gdsf->rho; //rho update
      double delta = std::max(0.0, gdsf->max_evicted_key - gdsf->clock_base); //delta는 항상 0이상 -> clock이 항상 증가
      gdsf->clock = gdsf->clock_base + rho * delta; //clock update
      //clock 갱신: key 칸에 MaxEvictedKey
      event_log_write(gdsf, "clock", -1, -1, -1, gdsf->max_evicted_key, gdsf->clock_base,
                      EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, rho, gdsf->clock, -1);
    }
  } else if (!hit) {
    //cache보다 큰 object: 넣지 않음
    event_log_write(gdsf, "bypass", req->obj_id, req->obj_size, -1, EV_NA, EV_NA, EV_NA, EV_NA,
                    EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, -1);
  }

  if (gdsf->rho_log != nullptr && cache->n_req % gdsf->rho_log_every == 0)
    rho_log_flush(gdsf, cache->n_req);

  DEBUG_ASSERT((int64_t)gdsf->pq.size() == cache->n_obj);
  DEBUG_ASSERT((int64_t)gdsf->pq_map.size() == cache->n_obj);

  return hit;
}

// ***********************************************************************
// ****                                                               ****
// ****       developer facing APIs (used by cache developer)         ****
// ****                                                               ****
// ***********************************************************************

/**
 * @brief find an object in the cache
 *
 * @param cache
 * @param req
 * @param update_cache whether to update the cache,
 *  if true, the object is promoted
 *  and if the object is expired, it is removed from the cache
 * @return the object or NULL if not found
 */
static cache_obj_t *GDSF_GCA_find(cache_t *cache, const request_t *req,
                              bool update_cache) {
  cache->n_req += 1;

  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
  cache_obj_t *obj = cache_find_base(cache, req, update_cache);
  /* this does not consider object size change */
  if (obj != nullptr && update_cache) {
    /* frequency is updated in cache_find_base */

    auto node = gdsf->pq_map[obj];
    gdsf->pq.erase(node);

    double pri =
        gdsf->clock + gca_value(gdsf, obj->freq, obj->obj_size);
    eviction::pq_node_type new_node = {obj, pri, cache->n_req};
    gdsf->pq.insert(new_node);
    gdsf->pq_map[obj] = new_node;
    gdsf->cur_req = cache->n_req;
    event_log_write(gdsf, "hit", obj->obj_id, obj->obj_size, obj->freq, pri, EV_NA,
                    gca_value(gdsf, obj->freq, obj->obj_size), EV_NA, EV_NA, EV_NA, EV_NA,
                    EV_NA, EV_NA, EV_NA, EV_NA, gdsf->clock, -1);
  }

  return obj;
}

static bool GDSF_GCA_can_insert(cache_t *cache, const request_t *req) {
  static __thread int64_t n_insert = 0, n_cannot_insert = 0;
  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
  if (cache->get_occupied_byte(cache) + req->obj_size <= cache->cache_size) {
    return true;
  }
  if (req->obj_size > cache->cache_size) {
    return false;
  }

  int64_t to_evict_size =
      req->obj_size - (cache->cache_size - cache->get_occupied_byte(cache));
  double pri = gdsf->clock + gca_value(gdsf, 1, req->obj_size); //넣을 object의 pri
  bool can_insert = true;
  auto iter = gdsf->pq.begin();

  int n_evict = 0;
  while (to_evict_size > 0) {
    assert(iter != gdsf->pq.end());
    assert(iter->obj->obj_id != req->obj_id);
    n_evict += 1;

    if (iter->priority > pri) {
      // the incoming object will be evicted so not insert it
      can_insert = false;
      break;
    }
    to_evict_size -= iter->obj->obj_size;
    iter++;
  }

  if (can_insert) {
    n_insert += 1;
  } else {
    n_cannot_insert += 1;
  }

  if ((n_insert + n_cannot_insert) % 100000 == 0) {
    if ((double)n_cannot_insert / (n_insert + n_cannot_insert) > 0.01)
      DEBUG("size %lld n_insert %lld, n_cannot_insert %lld, ratio %.2f\n",
            (long long)cache->cache_size, (long long)n_insert,
            (long long)n_cannot_insert,
            (double)n_cannot_insert / (n_insert + n_cannot_insert));
  }

  return can_insert;
}

/**
 * @brief insert an object into the cache,
 * update the hash table and cache metadata
 * this function assumes the cache has enough space
 * 일단 insert하고 넘치는 부분을 evict = gdsf와 같은 동작
 *
 * @param cache
 * @param req
 * @return the inserted object
 */
static cache_obj_t *GDSF_GCA_insert(cache_t *cache, const request_t *req) {
  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);

  // this does not affect insertion for most workloads unless object size is too
  // large however, when it have effect, it often increases miss ratio because a
  // list of small objects (with relatively large priority) will stop the
  // insertion of a large object, however, the newly requested large object is
  // likely to be more useful than the small objects if (!GDSF_GCA_can_insert(cache,
  // req)) return nullptr;

  cache_obj_t *obj = cache_insert_base(cache, req);
  DEBUG_ASSERT(obj != nullptr);
  obj->freq = gdsf->pending_freq0;  //보통 1, resurrect=1이고 ghost hit이었으면 freq0
  gdsf->pending_freq0 = 1;
  gdsf->bytes_admitted += obj->obj_size;

  double pri = gdsf->clock + gca_value(gdsf, obj->freq, obj->obj_size);
  eviction::pq_node_type new_node = {obj, pri, cache->n_req};
  auto r = gdsf->pq.insert(new_node);
  DEBUG_ASSERT(r.second);
  gdsf->pq_map[obj] = new_node;
  // cache에 insert 사건 // freq(=1), K, V, clock 기록
  event_log_write(gdsf, "insert", obj->obj_id, obj->obj_size, obj->freq, pri, EV_NA,
                  gca_value(gdsf, obj->freq, obj->obj_size), EV_NA, EV_NA, EV_NA, EV_NA, EV_NA,
                  EV_NA, EV_NA, EV_NA, gdsf->clock, -1);

  return obj;
}

/**
 * @brief find the object to be evicted
 * this function does not actually evict the object or update metadata
 * not all eviction algorithms support this function
 * because the eviction logic cannot be decoupled from finding eviction
 * candidate, so use assert(false) if you cannot support this function
 *
 * @param cache the cache
 * @return the object to be evicted
 */
static cache_obj_t *GDSF_GCA_to_evict(cache_t *cache, const request_t *req) {
  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
  eviction::pq_node_type p = gdsf->peek_lowest_score(); //victim 선택
  return p.obj;
}

/**
 * @brief evict an object from the cache
 * it needs to call cache_evict_base before returning
 * which updates some metadata such as n_obj, occupied size, and hash table
 *
 * @param cache
 * @param req not used
 * @param evicted_obj if not NULL, return the evicted object to caller
 */
static void GDSF_GCA_evict(cache_t *cache, const request_t *req) {
  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
  eviction::pq_node_type p = gdsf->pop_lowest_score(); //victim 축출
  // episode내에서 max_evicted_key 값, 총 victim 수 갱신
  if(gdsf->n_evicted_in_epi ==0 || p.priority > gdsf->max_evicted_key){
    gdsf-> max_evicted_key = p.priority;
  }
  gdsf->n_evicted_in_epi ++ ;

  double V = gca_value(gdsf, p.obj->freq, p.obj->obj_size);
  //(p.priority - gdsf->clock_base) = V(p) − (그 뒤로 clock이 오른 양)
  double slack = (p.priority - gdsf->clock_base)/ (V+1e-12);
  event_log_write(gdsf, "evict", p.obj->obj_id, p.obj->obj_size, p.obj->freq, p.priority,
                  gdsf->clock_base, V, slack, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA, EV_NA,
                  EV_NA, -1);
  ghost_insert(gdsf, p.obj->obj_id,
             {slack, gdsf->bytes_admitted, p.obj->obj_size, p.obj->freq});

  cache_remove_obj_base(cache, p.obj, true); //cache에서 삭제
}

static void GDSF_GCA_remove_obj(cache_t *cache, cache_obj_t *obj) {
  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
  gdsf->remove_obj(cache, obj);
}

/**
 * @brief remove an object from the cache
 * this is different from cache_evict because it is used to for user trigger
 * remove, and eviction is used by the cache to make space for new objects
 *
 * it needs to call cache_remove_obj_base before returning
 * which updates some metadata such as n_obj, occupied size, and hash table
 *
 * @param cache
 * @param obj_id
 * @return true if the object is removed, false if the object is not in the
 * cache
 */
static bool GDSF_GCA_remove(cache_t *cache, obj_id_t obj_id) {
  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
  return gdsf->remove(cache, obj_id);
}


#ifdef __cplusplus
}
#endif
