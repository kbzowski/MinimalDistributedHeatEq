/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2017 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */


// The program starts with the usual include files, all of which you should
// have seen before by now:
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/sparsity_tools.h>
// MPI STUFF
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>


#include <fstream>
#include <iostream>


namespace DistributedHE
{
	using namespace dealii;

	template<int dim>
		class HeatEquation
		{
		public:
			HeatEquation();
			void increment_time();
			void run();

		private:
			void setup_system();
			double solve_time_step();
			void create_grid();
			void output_results() const;
			void assemble_system();
			void make_dirichlet_boundary_conditions();

			MPI_Comm mpi_communicator;
			const unsigned int n_mpi_processes;
			const unsigned int this_mpi_process;

			ConditionalOStream pcout;

			parallel::shared::Triangulation<dim> triangulation;
			FE_Q<dim>            fe;
			DoFHandler<dim>      dof_handler;

			ConstraintMatrix     constraints;

			PETScWrappers::MPI::SparseMatrix system_matrix;

			Vector<double>                   solution;
			PETScWrappers::MPI::Vector       system_rhs;

			IndexSet locally_owned_dofs;
			IndexSet locally_relevant_dofs;
			std::vector<types::global_dof_index> local_dofs_per_process;
			unsigned int n_local_cells;

			double               time = 0;
			double               time_step = 60;
			unsigned int         timestep_number = 0;
			const double         theta = 0.5;
		}
		;


	template <int dim>
		void HeatEquation<dim>::increment_time()
		{
			timestep_number++;
			time += time_step;
		}

	template<int dim>
		void HeatEquation<dim>::create_grid()
		{
			const double size = 0.1;   	// meter
			GridGenerator::hyper_cube(triangulation, 0, size, true);
			triangulation.refine_global(5);
		}


	template<int dim>
		HeatEquation<dim>::HeatEquation()
			: mpi_communicator(MPI_COMM_WORLD)
			, n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
			, this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
			, pcout(std::cout, (this_mpi_process == 0))
			, triangulation(MPI_COMM_WORLD)
			, fe(1)
			, dof_handler(triangulation)
			, n_local_cells (numbers::invalid_unsigned_int)
		{
		
		}


	template<int dim>
		void HeatEquation<dim>::setup_system()
		{
			dof_handler.distribute_dofs(fe);
			locally_owned_dofs = dof_handler.locally_owned_dofs();
			DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
			n_local_cells  = GridTools::count_cells_with_subdomain_association(triangulation, triangulation.locally_owned_subdomain());
			local_dofs_per_process = dof_handler.n_locally_owned_dofs_per_processor();


			constraints.clear();
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			make_dirichlet_boundary_conditions();
			constraints.close();


			DynamicSparsityPattern dsp(locally_relevant_dofs);
			DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, /*keep_constrained_dofs = */ false);
			SparsityTools::distribute_sparsity_pattern(dsp,
				local_dofs_per_process,
				mpi_communicator,
				locally_relevant_dofs);



				
			system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
			

			if (timestep_number == 0) 
			{
				solution.reinit(dof_handler.n_dofs()); 
				solution = 800;
			}

			system_rhs.reinit(locally_owned_dofs, mpi_communicator);
		}


	template<int dim>
		void HeatEquation<dim>::output_results() const
		{
			std::string filename = "solution-" + Utilities::int_to_string(timestep_number, 4)
			                       + "." + Utilities::int_to_string(this_mpi_process, 3)
			                       + ".vtu";

			std::ofstream output(filename.c_str());
			DataOut<dim> data_out;
			data_out.attach_dof_handler(dof_handler);
			data_out.add_data_vector(solution, "Temperature");
			std::vector<unsigned int> partition_int(triangulation.n_active_cells());
			GridTools::get_subdomain_association(triangulation, partition_int);
			const Vector<double> partitioning(partition_int.begin(), partition_int.end());
			data_out.add_data_vector(partitioning, "Partitioning");
			data_out.build_patches();
			data_out.write_vtu(output);

			if (this_mpi_process == 0)
			{
				std::vector<std::string> filenames;
				for (unsigned int i = 0; i < n_mpi_processes; ++i)
					filenames.push_back("solution-" + Utilities::int_to_string(timestep_number, 4)
					                     + "." + Utilities::int_to_string(i, 3)
					                     + ".vtu");
				
				const std::string pvtu_master_filename = ("solution-" +
				                        Utilities::int_to_string(timestep_number, 4) +
				                        ".pvtu");
				std::ofstream pvtu_master(pvtu_master_filename.c_str());
				data_out.write_pvtu_record(pvtu_master, filenames);
				static std::vector<std::pair<double, std::string> > times_and_names;
				times_and_names.push_back(std::pair<double, std::string> (time, pvtu_master_filename));
				std::ofstream pvd_output("solution.pvd");
				DataOutBase::write_pvd_record(pvd_output, times_and_names);
			}

			
		}


	template<int dim>
		void HeatEquation<dim>::assemble_system()
		{
			double temp_env_top = 20;
			double convection_top = 0;
			double temp_env_right = 20;
			double convection_right = 0;

			QGauss<dim> quadrature_formula(2);
			QGauss<dim - 1> face_quadrature_formula(3);
			
			QTrapez<3> quadrature;

			FEValues<dim> cell_fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points);
			FEFaceValues<dim> face_fe_values(fe, face_quadrature_formula, update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);
			FEValues<dim> fe_values(fe, quadrature, update_values);

			const unsigned int dofs_per_cell = fe.dofs_per_cell;

			const unsigned int cell_quadrature_points = quadrature_formula.size();
			const unsigned int face_q_pts = face_quadrature_formula.size();

			// Auxiliary matrices
			FullMatrix<double> cell_h(dofs_per_cell, dofs_per_cell);
			FullMatrix<double> cell_hq(dofs_per_cell, dofs_per_cell);
			FullMatrix<double> cell_c(dofs_per_cell, dofs_per_cell);
			Vector<double> cell_p(dofs_per_cell);

			// Resulting matrices
			FullMatrix<double> cell_a(dofs_per_cell, dofs_per_cell);
			Vector<double> cell_rhs(dofs_per_cell);

			typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();

			const double cond = 90;
			const double cp = 490;
			const double ro = 7820;
			double alpha = 0;
			double tenv = 0;
			
			for (; cell != endc; ++cell) {
				if (cell->is_locally_owned())
				{
					
					cell_fe_values.reinit(cell);
					fe_values.reinit(cell);

					cell_h = 0;
					cell_c = 0;
					cell_p = 0;
					cell_hq = 0;


					// Reinit matrices by 0
					cell_a = 0;
					cell_rhs = 0;

					for (unsigned int i = 0; i < dofs_per_cell; ++i) {
						for (unsigned int j = 0; j < dofs_per_cell; ++j) {
							for (unsigned int q_point = 0; q_point < cell_quadrature_points; ++q_point) {

								double dn_dx = cell_fe_values.shape_grad(i, q_point)[0];
								double dk_dx = cell_fe_values.shape_grad(j, q_point)[0];

								double dn_dy = cell_fe_values.shape_grad(i, q_point)[1];
								double dk_dy = cell_fe_values.shape_grad(j, q_point)[1];

								double n = cell_fe_values.shape_value(i, q_point);
								double k = cell_fe_values.shape_value(j, q_point);

								double jdet = cell_fe_values.JxW(q_point);

								cell_h(i, j) += cond * jdet * (dn_dx * dk_dx + dn_dy * dk_dy);
								cell_c(i, j) += cp * ro * n * k * jdet;
							}
						}
					}


					// Apply Neumann boundary condition
					for(unsigned int face = 0 ; face < GeometryInfo<dim>::faces_per_cell ; ++face) {
						if (cell->face(face)->at_boundary() && (cell->face(face)->boundary_id() == 2 || cell->face(face)->boundary_id() == 3)) {

							if (cell->face(face)->boundary_id() == 2) {
								alpha = convection_top;
								tenv = temp_env_top;
							}
							else if (cell->face(face)->boundary_id() == 3) {
								alpha = convection_right;
								tenv = temp_env_right;
							}

							//Point<dim> debugpt = cell->face(face)->center();
							face_fe_values.reinit(cell, face);

							for (unsigned int q = 0; q < face_q_pts; ++q) {
								double jdet = face_fe_values.JxW(q);
								for (unsigned int ii = 0; ii < dofs_per_cell; ++ii) {
									double n = face_fe_values.shape_value(ii, q);
									cell_p(ii) += alpha * tenv * n * jdet;

									for (unsigned int jj = 0; jj < dofs_per_cell; jj++) {
										double k = face_fe_values.shape_value(jj, q);
										cell_hq(ii, jj) += alpha * (n * k) * jdet;

									}

								}
							}
						}
					}


					// Integration over time
					// values from previous time step

					std::vector<double> local_values(dofs_per_cell);
					fe_values.get_function_values(solution, local_values);

					//	std::vector<double> local_values(dofs_per_cell);
					//	cell_fe_values.get_function_values(solution, local_values);
					//					
					//	Vector<double> local_values(dofs_per_cell);
					//	cell->get_dof_values(solution, local_values);

										for(unsigned int i = 0 ; i < dofs_per_cell ; ++i) {
						for (unsigned int j = 0; j < dofs_per_cell; ++j) {
							// Galerkin
							double t0 = local_values[j];
//							cell_a(i, j) += 2.0 * cell_h(i,j) + (3.0/time_step) * cell_c(i,j) + cell_hq(i,j);	
//							cell_rhs(i) += (-cell_h(i,j) + (3.0/time_step) * cell_c(i,j)) * t0;

							// Crank-Nicolson
							//cell_a(i, j) += cell_h(i, j) + 2.0*cell_c(i, j) / time_step + cell_hq(i, j);	
							//cell_rhs(i) += (-cell_h(i, j) + 2.0*cell_c(i, j) / time_step) * t0;				

							// Euler
							cell_a(i, j) += cell_h(i, j) + cell_hq(i, j) + cell_c(i, j) / time_step;
							cell_rhs(i) += (cell_c(i, j) / time_step) * t0;
						}

						// Galerkin
//						cell_rhs(i) += 3.0 * cell_p(i);

						// Crank-Nicolson
						//cell_rhs(i) += 2.0 * cell_p(i);

						//Euler
						cell_rhs(i) += cell_p(i);
					}

					
					// Aggregation system of equation
					 std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
					cell->get_dof_indices(local_dof_indices);
					constraints.distribute_local_to_global(cell_a, cell_rhs, local_dof_indices, system_matrix, system_rhs);
				}
			}

			system_matrix.compress(VectorOperation::add);
			system_rhs.compress(VectorOperation::add);

		}

	template <int dim>
	void HeatEquation<dim>::make_dirichlet_boundary_conditions()
	{
		VectorTools::interpolate_boundary_values(dof_handler, 2, ConstantFunction<dim>(20), constraints);
		VectorTools::interpolate_boundary_values(dof_handler, 3, ConstantFunction<dim>(20), constraints);
	}

	template<int dim>
		double HeatEquation<dim>::solve_time_step()
		{
			PETScWrappers::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);
			distributed_solution = solution;

			SolverControl solver_control(dof_handler.n_dofs(), 1e-8*system_rhs.l2_norm());
			PETScWrappers::SolverCG cg(solver_control, mpi_communicator);
			PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
			cg.solve(system_matrix, distributed_solution, system_rhs, preconditioner);
			constraints.distribute(distributed_solution);
			solution = distributed_solution;
			
			return solver_control.last_step();
		}


	template<int dim>
		void HeatEquation<dim>::run()
		{
			for (unsigned int time = 0; time < 10; ++time)
			{
				pcout << "Time " << time << ':' << std::endl;
				if (time == 0)
				{
					create_grid();
				}

				pcout << "   Number of active cells:       "
				      << triangulation.n_active_cells()
				      << std::endl;
				setup_system();
				pcout << "   Number of degrees of freedom: "
				      << dof_handler.n_dofs()
				      << " (by partition:";
				for (unsigned int p = 0; p < n_mpi_processes; ++p)
					pcout << (p == 0 ? ' ' : '+')
					      << (DoFTools::
					          count_dofs_with_subdomain_association(dof_handler, p));

				pcout << ")" << std::endl;
				pcout << "   Assembling." << std::endl;
				assemble_system();
				pcout << "   Solving." << std::endl;
				const unsigned int n_iterations = solve_time_step();
				pcout << "   Solver converged in " << n_iterations
				      << " iterations." << std::endl;
				output_results();
				increment_time();
			}
		}
}
	

int main(int argc, char **argv)
{
	try
	{
		using namespace dealii;
		using namespace DistributedHE;

		Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
		HeatEquation<3> heat_equation_solver;
		heat_equation_solver.run();

	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl
		          << "----------------------------------------------------"
		          << std::endl;
		std::cerr << "Exception on processing: " << std::endl << exc.what()
		          << std::endl << "Aborting!" << std::endl
		          << "----------------------------------------------------"
		          << std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl
		          << "----------------------------------------------------"
		          << std::endl;
		std::cerr << "Unknown exception!" << std::endl << "Aborting!"
		          << std::endl
		          << "----------------------------------------------------"
		          << std::endl;
		return 1;
	}

	return 0;
}