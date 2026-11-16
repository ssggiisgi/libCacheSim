# ==============================================================================
# pinsim: replay a cloud OLAP request log with pin/unpin (GDSF-GCA, pin=1)
# ==============================================================================

add_executable(pinsim main.cpp)
target_include_directories(pinsim PRIVATE ${libCacheSim_include_dir} ${GLib_INCLUDE_DIRS})
target_link_libraries(pinsim ${PROJECT_NAME})
install(TARGETS pinsim RUNTIME DESTINATION bin)
