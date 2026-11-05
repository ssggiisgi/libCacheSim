/* GDSize: greedy dual size (Cao and Irani, USITS 1997)
 *
 * priority H(p) = L + cost(p) / size(p), with cost = 1, and L is the priority
 * of the last evicted object. Unlike GDSF there is no frequency term: a hit
 * resets H(p) to L + cost / size. Everything else (insert then evict, ties
 * broken by FIFO, the 1e6 scale) follows GDSF.cpp, so GDSize and GDSF differ
 * only in the frequency term.
 */

#include <cassert>

#include "abstractRank.hpp"

namespace eviction {
class GDSize : public abstractRank {
 public:
  GDSize() = default;

  double pri_last_evict = 0.0;
};
}  // namespace eviction

#ifdef __cplusplus
extern "C" {
#endif

// ***********************************************************************
// ****                                                               ****
// ****                   function declarations                       ****
// ****                                                               ****
// ***********************************************************************

cache_t *GDSize_init(const common_cache_params_t ccache_params,
                     const char *cache_specific_params);
static void GDSize_free(cache_t *cache);
static bool GDSize_get(cache_t *cache, const request_t *req);

static cache_obj_t *GDSize_find(cache_t *cache, const request_t *req,
                                bool update_cache);
static cache_obj_t *GDSize_insert(cache_t *cache, const request_t *req);
static cache_obj_t *GDSize_to_evict(cache_t *cache, const request_t *req);
static void GDSize_evict(cache_t *cache, const request_t *req);

static bool GDSize_remove(cache_t *cache, obj_id_t obj_id);

// ***********************************************************************
// ****                                                               ****
// ****                   end user facing functions                   ****
// ****                                                               ****
// ****                       init, free, get                         ****
// ***********************************************************************

/**
 * @brief initialize the cache
 *
 * @param ccache_params some common cache parameters
 * @param cache_specific_params cache specific parameters, see parse_params
 * function or use -e "print" with the cachesim binary
 */
cache_t *GDSize_init(const common_cache_params_t ccache_params,
                     const char *cache_specific_params) {
  cache_t *cache =
      cache_struct_init("GDSize", ccache_params, cache_specific_params);
  cache->eviction_params = reinterpret_cast<void *>(new eviction::GDSize);

  cache->cache_init = GDSize_init;
  cache->cache_free = GDSize_free;
  cache->get = GDSize_get;
  cache->find = GDSize_find;
  cache->insert = GDSize_insert;
  cache->evict = GDSize_evict;
  cache->to_evict = GDSize_to_evict;
  cache->remove = GDSize_remove;

  if (ccache_params.consider_obj_metadata) {
    // priority
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
static void GDSize_free(cache_t *cache) {
  delete reinterpret_cast<eviction::GDSize *>(cache->eviction_params);
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
 *    insert the object
 *    while the cache is over capacity:
 *        evict
 *    return false
 * ```
 *
 * @param cache
 * @param req
 * @return true if cache hit, false if cache miss
 */
static bool GDSize_get(cache_t *cache, const request_t *req) {
  auto *gds = reinterpret_cast<eviction::GDSize *>(cache->eviction_params);
  cache_obj_t *obj = cache->find(cache, req, true);
  bool hit = (obj != NULL);

  if (!hit && cache->can_insert(cache, req)) {
    cache->insert(cache, req);
    while (cache->get_occupied_byte(cache) > cache->cache_size) {
      cache->evict(cache, req);
    }
  }

  DEBUG_ASSERT((int64_t)gds->pq.size() == cache->n_obj);
  DEBUG_ASSERT((int64_t)gds->pq_map.size() == cache->n_obj);

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
static cache_obj_t *GDSize_find(cache_t *cache, const request_t *req,
                                bool update_cache) {
  cache->n_req += 1;

  auto *gds = reinterpret_cast<eviction::GDSize *>(cache->eviction_params);
  cache_obj_t *obj = cache_find_base(cache, req, update_cache);
  /* this does not consider object size change */
  if (obj != nullptr && update_cache) {
    auto node = gds->pq_map[obj];
    gds->pq.erase(node);

    // a hit restores H to L + cost / size; frequency is not used
    double pri = gds->pri_last_evict + 1.0e6 / obj->obj_size;
    eviction::pq_node_type new_node = {obj, pri, cache->n_req};
    gds->pq.insert(new_node);
    gds->pq_map[obj] = new_node;
  }

  return obj;
}

/**
 * @brief insert an object into the cache,
 * update the hash table and cache metadata
 * the cache may go over capacity, GDSize_get evicts afterwards (as GDSF)
 *
 * @param cache
 * @param req
 * @return the inserted object
 */
static cache_obj_t *GDSize_insert(cache_t *cache, const request_t *req) {
  auto *gds = reinterpret_cast<eviction::GDSize *>(cache->eviction_params);

  cache_obj_t *obj = cache_insert_base(cache, req);
  DEBUG_ASSERT(obj != nullptr);

  double pri = gds->pri_last_evict + 1.0e6 / obj->obj_size;
  eviction::pq_node_type new_node = {obj, pri, cache->n_req};
  auto r = gds->pq.insert(new_node);
  DEBUG_ASSERT(r.second);
  gds->pq_map[obj] = new_node;

  return obj;
}

/**
 * @brief find the object to be evicted
 * this function does not actually evict the object or update metadata
 *
 * @param cache the cache
 * @return the object to be evicted
 */
static cache_obj_t *GDSize_to_evict(cache_t *cache, const request_t *req) {
  auto *gds = reinterpret_cast<eviction::GDSize *>(cache->eviction_params);
  eviction::pq_node_type p = gds->peek_lowest_score();

  return p.obj;
}

/**
 * @brief evict an object from the cache
 * it needs to call cache_evict_base before returning
 * which updates some metadata such as n_obj, occupied size, and hash table
 *
 * @param cache
 * @param req not used
 */
static void GDSize_evict(cache_t *cache, const request_t *req) {
  auto *gds = reinterpret_cast<eviction::GDSize *>(cache->eviction_params);
  eviction::pq_node_type p = gds->pop_lowest_score();
  cache_obj_t *obj = p.obj;

  gds->pri_last_evict = p.priority;  // L
  cache_remove_obj_base(cache, obj, true);
}

/**
 * @brief remove an object from the cache
 * this is different from cache_evict because it is used to for user trigger
 * remove, and eviction is used by the cache to make space for new objects
 *
 * @param cache
 * @param obj_id
 * @return true if the object is removed, false if the object is not in the
 * cache
 */
static bool GDSize_remove(cache_t *cache, obj_id_t obj_id) {
  auto *gds = reinterpret_cast<eviction::GDSize *>(cache->eviction_params);
  return gds->remove(cache, obj_id);
}

#ifdef __cplusplus
}
#endif
