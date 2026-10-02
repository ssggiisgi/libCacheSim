/* GDSF_GCA: greedy dual frequency size */

#include <cassert>
#include <list>
#include <algorithm>
#include "abstractRank.hpp"

namespace eviction {

//ghost entry
struct ghost_entry {
  double slack_norm;
  int64_t bytes_admitted_at_evict;
  int64_t obj_size;
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
  double eta = 0.05, w_mem = 1.0, w_burst = 0.25, w_res = 1.0;
  double rho_min = 0.01, rho_max = 4.0, ghost_ratio = 1.0;
  //eta : 학습률


  //ghost list
  std::list<obj_id_t> ghost_lru; //front에 넣고 back에서 축출
  std::unordered_map<obj_id_t, std::pair<ghost_entry, std::list<obj_id_t>::iterator>> ghost_map;
  int64_t ghost_bytes = 0;
  int64_t ghost_capacity = 0; // cache_size*ghost_ratio
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
//파라미터 파싱//
///////////////////////////////////////////////////
static void GDSF_GCA_parse_params(cache_t *cache, const char *params) {
  auto *gdsf = reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
  //기본값
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
    else if (strcasecmp(key, "rho-min") == 0)     gdsf->rho_min = strtod(value, nullptr);
    else if (strcasecmp(key, "rho-max") == 0)     gdsf->rho_max = strtod(value, nullptr);
    else if (strcasecmp(key, "ghost-ratio") == 0) gdsf->ghost_ratio = strtod(value, nullptr);
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
//ghost insert
//////////////////////////////
static void ghost_insert(eviction::GDSF_GCA *g, obj_id_t id, const eviction::ghost_entry &e){
  //이미 ghost에 있으면 먼저 제거 <- 이런 경우가 있을 수가 있나????
  auto old = g->ghost_map.find(id);
  if (old != g->ghost_map.end()) ghost_erase(g, old);

  g->ghost_lru.push_front(id);
  g->ghost_map[id] = {e, g->ghost_lru.begin()};
  g->ghost_bytes += e.obj_size;
  //ghost list 넘치면 축출
  while (g->ghost_bytes > g->ghost_capacity && !g->ghost_lru.empty()){
    ghost_erase(g, g->ghost_map.find(g->ghost_lru.back()));
  }
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
  delete reinterpret_cast<eviction::GDSF_GCA *>(cache->eviction_params);
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

  //cache miss
  if (!hit && cache->can_insert(cache, req)) {
    //ghost look up
    auto it = gdsf->ghost_map.find(req->obj_id);
     //ghost에서 발견됨 -> rho 업데이트
    if (it != gdsf->ghost_map.end()){
      const eviction::ghost_entry &e = it->second.first;
      double turns = (double)(gdsf->bytes_admitted - e.bytes_admitted_at_evict)/ cache->cache_size;
      if (gdsf->learn){
        double s = std::min(1.0, std::max(0.0, e.slack_norm));
        double g = 2.0 * s -1.0; //s가 0이면 -1(aging 늦추기), s가 1이면 1(aging 강하게)
        double w = exp(-turns / gdsf->w_mem); //금방 재요청된건지 아닌지 표현
        if (g>0) g *= exp(-turns / gdsf->w_burst); //g>0일때 한참 뒤에 재요청된 item의 영향 작게
        gdsf->log_rho += gdsf->eta *w *g;
        gdsf->log_rho = std::clamp(gdsf->log_rho, log(gdsf->rho_min), log(gdsf->rho_max));
      }
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
    }
  }

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
        gdsf->clock + (double)(obj->freq) * 1.0e6 / obj->obj_size;
    eviction::pq_node_type new_node = {obj, pri, cache->n_req};
    gdsf->pq.insert(new_node);
    gdsf->pq_map[obj] = new_node;
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
  double pri = gdsf->clock + 1.0e6 / req->obj_size;
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
 * eviction should be
 * performed before calling this function
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
  obj->freq = 1; // resurrection 구현해야 함!!
  gdsf->bytes_admitted += obj->obj_size;

  double pri = gdsf->clock + 1.0e6 / obj->obj_size;
  eviction::pq_node_type new_node = {obj, pri, cache->n_req};
  auto r = gdsf->pq.insert(new_node);
  DEBUG_ASSERT(r.second);
  gdsf->pq_map[obj] = new_node;

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

  double V = (double)p.obj->freq * 1.0e6 / p.obj->obj_size;
  //(p.priority - gdsf->clock_base) = V(p) − (그 뒤로 clock이 오른 양)
  double slack = (p.priority - gdsf->clock_base)/ (V+1e-12);
  ghost_insert(gdsf, p.obj->obj_id,
             {slack, gdsf->bytes_admitted, p.obj->obj_size});

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
