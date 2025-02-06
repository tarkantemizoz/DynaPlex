
cmake_minimum_required (VERSION 3.21)

if(MPI_FOUND)
add_subdirectory(mpi_tests)
endif()

add_subdirectory(lostsales_paper_results)
add_subdirectory(perishables_paper_results)
add_subdirectory(generally_capable_lost_sales_agent)
add_subdirectory(randomleadtimes_paper_results)
add_subdirectory(multi_item_sla)
add_subdirectory(driver_assignment)