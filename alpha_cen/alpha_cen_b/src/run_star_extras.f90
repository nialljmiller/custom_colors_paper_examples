! ***********************************************************************
!
!   Copyright (C) 2011  Bill Paxton & The MESA Team
!
!   This program is free software: you can redistribute it and/or modify
!   it under the terms of the GNU Lesser General Public License
!   as published by the Free Software Foundation,
!   either version 3 of the License, or (at your option) any later version.
!
!   This program is distributed in the hope that it will be useful,
!   but WITHOUT ANY WARRANTY; without even the implied warranty of
!   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!   See the GNU Lesser General Public License for more details.
!
!   You should have received a copy of the GNU Lesser General Public License
!   along with this program. If not, see <https://www.gnu.org/licenses/>.
!
! ***********************************************************************

      module run_star_extras

      use star_lib
      use star_def
      use const_def
      use math_lib

      implicit none

      logical, save :: colors_cache_valid = .false.
      integer, save :: colors_cache_star_id = -1
      integer, save :: colors_cache_model_number = -1
      real(dp), save :: colors_cache_teff = 0d0
      real(dp), save :: colors_cache_logg = 0d0
      real(dp), save :: colors_cache_radius = 0d0
      real(dp), save :: colors_cache_metallicity = 0d0
      character(len=80), allocatable, save :: colors_cache_names(:)
      real(dp), allocatable, save :: colors_cache_values(:)


      ! these routines are called by the standard run_star check_model
      contains

      subroutine extras_controls(id, ierr)
         use astero_def, only: star_astero_procs
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         s% extras_startup => extras_startup
         s% extras_check_model => extras_check_model
         s% extras_finish_step => extras_finish_step
         s% extras_after_evolve => extras_after_evolve
         s% how_many_extra_history_columns => how_many_extra_history_columns
         s% data_for_extra_history_columns => data_for_extra_history_columns
         s% how_many_extra_profile_columns => how_many_extra_profile_columns
         s% data_for_extra_profile_columns => data_for_extra_profile_columns
         include 'set_star_astero_procs.inc'
      end subroutine extras_controls


      subroutine set_constraint_value(id, name, val, ierr)  ! called from star_astero code
         use astero_def, only: Z_div_X_solar

         integer, intent(in) :: id
         character(len=strlen), intent(in) :: name
         real(dp), intent(out) :: val
         integer, intent(out) :: ierr

         type (star_info), pointer :: s
         integer :: i
         real(dp) :: X, Y, Z, B_mag, V_mag, Ks_mag
         ! constraints are predefined in the simplex_search_data.
         ! this routine's job is to assign those variables to current value in the model.
         ! it is called whenever a new value of chi2 is calculated.
         ! only necessary to set the constraints you are actually using.
         ierr = 0

         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         X = max(s% surface_h1, 1d-10)
         Y = s% surface_he3 + s% surface_he4
         Z = max(1d-99, min(1d0, 1-X-Y))

         select case (name)
            ! for custom constraints, create a case with the name of your constraint e.g.
            ! case ('delta_Pg')
            !    val = s% delta_Pg
            ! fall back to history column if user doesn't define name
            case ('M_H')
               val = log10(Z/X/Z_div_X_solar)

            case ('surface_Z_div_X')
               val = Z/X

            case ('surface_He')
               val = Y

            case ('Rcz')
               do i = 1, s% nz-1  ! locate bottom of solar convective zone
                  if (s% mixing_type(i+1) /= convective_mixing &
                        .and. s% mixing_type(i) == convective_mixing) then
                     if (s% r(i+1) > 0.25d0*Rsun .and. s% r(i) < 0.9d0*Rsun) then
                        val = s% r(i)/Rsun
                        exit
                     end if
                  end if
               end do

            case ('M_V')
               call current_color_value(s, 'V', val, ierr)

            case ('B_minus_V')
               call current_color_value(s, 'B', B_mag, ierr)
               if (ierr /= 0) return
               call current_color_value(s, 'V', V_mag, ierr)
               if (ierr /= 0) return
               val = B_mag - V_mag

            case ('V_minus_Ks')
               call current_color_value(s, 'V', V_mag, ierr)
               if (ierr /= 0) return
               call current_color_value(s, 'Ks', Ks_mag, ierr)
               if (ierr /= 0) return
               val = V_mag - Ks_mag

            case default
               val = star_get_history_output(s, name, ierr)
               if (ierr /= 0) call mesa_error(__FILE__, __LINE__)
         end select

      end subroutine set_constraint_value


      subroutine current_color_value(s, color_name, val, ierr)
         type (star_info), pointer :: s
         character(len=*), intent(in) :: color_name
         real(dp), intent(out) :: val
         integer, intent(out) :: ierr

         integer :: i

         val = -1d99
         call update_colors_cache(s, ierr)
         if (ierr /= 0) return

         do i = 1, size(colors_cache_names)
            if (trim(colors_cache_names(i)) == trim(color_name)) then
               val = colors_cache_values(i)
               return
            end if
         end do

         write(*,*) 'Colors constraint not found: ', trim(color_name)
         ierr = -1
      end subroutine current_color_value


      subroutine update_colors_cache(s, ierr)
         use colors_def, only: Colors_General_Info
         use colors_lib, only: colors_ptr, how_many_colors_history_columns, &
            data_for_colors_history_columns

         type (star_info), pointer :: s
         integer, intent(out) :: ierr

         type (Colors_General_Info), pointer :: colors_settings
         integer :: k, n
         real(dp) :: teff, logg, radius, metallicity
         real(dp) :: min_metallicity, max_metallicity

         ierr = 0
         call colors_ptr(s% colors_handle, colors_settings, ierr)
         if (ierr /= 0) return

         if (.not. colors_settings% use_colors) then
            write(*,*) 'Colors constraints require use_colors = .true.'
            ierr = -1
            return
         end if

         if (.not. allocated(colors_settings% lu_meta)) then
            write(*,*) 'Colors atmosphere metallicity grid is not loaded.'
            ierr = -1
            return
         end if

         if (colors_settings% z_over_x_ref <= 0d0) then
            write(*,*) 'Colors z_over_x_ref must be positive.'
            ierr = -1
            return
         end if

         k = s% photosphere_cell_k
         if (k < 1 .or. k > s% nz) then
            write(*,*) 'Invalid photosphere_cell_k for Colors constraint: ', k
            ierr = -1
            return
         end if

         ! Match the inputs used by star/private/history.f90 in this
         ! MESA revision so the fitted quantities and Colors history
         ! columns describe exactly the same stellar surface.
         teff = s% Teff
         logg = s% photosphere_logg
         radius = s% photosphere_r*Rsun

         min_metallicity = minval(colors_settings% lu_meta)
         max_metallicity = maxval(colors_settings% lu_meta)
         if (s% X(k) > 0d0 .and. s% Z(k) > 0d0) then
            metallicity = log10((s% Z(k)/s% X(k)) / &
               colors_settings% z_over_x_ref)
            metallicity = max(min_metallicity, &
               min(max_metallicity, metallicity))
         else
            metallicity = min_metallicity
         end if

         if (colors_cache_valid .and. &
               colors_cache_star_id == s% id .and. &
               colors_cache_model_number == s% model_number .and. &
               colors_cache_teff == teff .and. &
               colors_cache_logg == logg .and. &
               colors_cache_radius == radius .and. &
               colors_cache_metallicity == metallicity) return

         n = how_many_colors_history_columns(s% colors_handle)
         if (n <= 0) then
            write(*,*) 'Colors returned no history columns.'
            ierr = -1
            return
         end if

         if (allocated(colors_cache_names)) then
            if (size(colors_cache_names) /= n) then
               deallocate(colors_cache_names, colors_cache_values)
            end if
         end if
         if (.not. allocated(colors_cache_names)) then
            allocate(colors_cache_names(n), colors_cache_values(n), stat=ierr)
            if (ierr /= 0) return
         end if

         colors_cache_names = 'unknown'
         colors_cache_values = -1d99
         call data_for_colors_history_columns(teff, logg, radius, metallicity, &
            s% model_number, s% colors_handle, n, colors_cache_names, &
            colors_cache_values, ierr)
         if (ierr /= 0) then
            colors_cache_valid = .false.
            return
         end if

         colors_cache_star_id = s% id
         colors_cache_model_number = s% model_number
         colors_cache_teff = teff
         colors_cache_logg = logg
         colors_cache_radius = radius
         colors_cache_metallicity = metallicity
         colors_cache_valid = .true.
      end subroutine update_colors_cache


      subroutine set_param(id, name, val, ierr)  ! called from star_astero code
         use astero_def, only: f0_ov_div_f_ov, Y_frac_he3, Z_div_X_solar, &
            Y_depends_on_Z, dYdZ, Y0

         integer, intent(in) :: id
         character(len=strlen), intent(in) :: name  ! which of param's will be set
         real(dp), intent(in) :: val
         integer, intent(out) :: ierr
         type (star_info), pointer :: s

         real(dp) :: X, Y, Z_div_X, c

         ierr = 0

         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         select case (name)
            case ('initial_mass')
               s% job% new_mass = val
               s% job% relax_initial_mass = .true.
               s% initial_mass = val
            case ('initial_Y')
               s% job% initial_he3 = Y_frac_he3*val
               s% job% initial_he4 = val - s% job% initial_he3
               s% job% set_uniform_initial_composition = .true.
            case ('initial_FeH')
               ! this only works because initial_Y is an earlier parameter than FeH
               ! so set first and we can use that value to infer Z and X
               Z_div_X = Z_div_X_solar*exp10(val)  ! (Z/X) = (Z/X)sun * 10^[Fe/H]

               if (Y_depends_on_Z) then
                  c = 1d0 + Z_div_X*(1d0 + dYdZ)
                  X = (1d0 - Y0)/c
                  Y = (Y0 + Z_div_X*(dYdZ + Y0))/c

                  s% job% initial_he3 = Y_frac_he3*Y
                  s% job% initial_he4 = Y - s% job% initial_he3
               else
                  Y = s% job% initial_he3/Y_frac_he3  ! get Y, which we know has been set
                  X = (1d0 - Y)/(1d0 + Z_div_X)  ! X = (1-Y)/(1 + (Z/X))
               end if

               s% job% initial_h1 = X
               s% job% initial_h2 = 0
               s% job% set_uniform_initial_composition = .true.
            case ('alpha')
               s% mixing_length_alpha = val
            case ('f_ov')
               if (val > 0._dp) then
                  s% overshoot_scheme(1) = 'exponential'
                  s% overshoot_zone_type(1) = 'any'
                  s% overshoot_zone_loc(1) = 'any'
                  s% overshoot_bdy_loc(1) = 'any'
                  s% overshoot_f(1) = val
                  s% overshoot_f0(1) = f0_ov_div_f_ov*val
               else
                  s% overshoot_scheme(1) = ''
                  s% overshoot_zone_type(1) = ''
                  s% overshoot_zone_loc(1) = ''
                  s% overshoot_bdy_loc(1) = ''
                  s% overshoot_f(1) = 0d0
                  s% overshoot_f0(1) = 0d0
               end if
            case default
               ierr = -1
               write(*,*) 'invalid name in set_param', name
            end select

      end subroutine set_param


      subroutine extras_startup(id, restart, ierr)
         integer, intent(in) :: id
         logical, intent(in) :: restart
         integer, intent(out) :: ierr
         ierr = 0
         ! astero may run several tracks in one executable.  Do not allow
         ! a cached Colors result from the preceding track to be reused.
         colors_cache_valid = .false.
      end subroutine extras_startup


      subroutine extras_after_evolve(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         ierr = 0
      end subroutine extras_after_evolve


      integer function extras_check_model(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         extras_check_model = keep_going
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         include 'formats'

         ! if you want to check multiple conditions, it can be useful
         ! to set a different termination code depending on which
         ! condition was triggered.  MESA provides 9 customizable
         ! termination codes, named t_xtra1 .. t_xtra9.  You can
         ! customize the messages that will be printed upon exit by
         ! setting the corresponding termination_code_str value.
         ! termination_code_str(t_xtra1) = 'my termination condition'

         ! by default, indicate where (in the code) MESA terminated
         if (extras_check_model == terminate) s% termination_code = t_extras_check_model
      end function extras_check_model


      integer function how_many_extra_history_columns(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_history_columns = 2
      end function how_many_extra_history_columns


      subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         names(1) = 'fit_M_V'
         call current_color_value(s, 'V', vals(1), ierr)
         if (ierr /= 0) return

         names(2) = 'fit_B_minus_V'
         call current_color_value(s, 'B', vals(2), ierr)
         if (ierr /= 0) return
         vals(2) = vals(2) - vals(1)
      end subroutine data_for_extra_history_columns


      integer function how_many_extra_profile_columns(id)
         use star_def, only: star_info
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_profile_columns = 0
      end function how_many_extra_profile_columns


      subroutine data_for_extra_profile_columns(id, n, nz, names, vals, ierr)
         use star_def, only: star_info, maxlen_profile_column_name
         use const_def, only: dp
         integer, intent(in) :: id, n, nz
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(nz,n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         integer :: k
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
      end subroutine data_for_extra_profile_columns


	integer function extras_finish_step(id)
	   use astero_def, only: chi2, &
	      chi2_limit_for_small_timesteps, &
	      chi2_limit_for_smaller_timesteps, &
	      chi2_limit_for_smallest_timesteps, &
	      max_yrs_dt_chi2_small_limit, &
	      max_yrs_dt_chi2_smaller_limit, &
	      max_yrs_dt_chi2_smallest_limit

	   integer, intent(in) :: id
	   integer :: ierr
	   real(dp) :: dt_cap_yr
	   type(star_info), pointer :: s

	   extras_finish_step = terminate

	   ierr = 0
	   call star_ptr(id, s, ierr)
	   if (ierr /= 0) return

	   dt_cap_yr = -1d0

	   if (chi2 >= 0d0) then
	      if (chi2_limit_for_smallest_timesteps >= 0d0 .and. &
		    chi2 <= chi2_limit_for_smallest_timesteps) then
		 dt_cap_yr = max_yrs_dt_chi2_smallest_limit

	      else if (chi2_limit_for_smaller_timesteps >= 0d0 .and. &
		    chi2 <= chi2_limit_for_smaller_timesteps) then
		 dt_cap_yr = max_yrs_dt_chi2_smaller_limit

	      else if (chi2_limit_for_small_timesteps >= 0d0 .and. &
		    chi2 <= chi2_limit_for_small_timesteps) then
		 dt_cap_yr = max_yrs_dt_chi2_small_limit
	      end if
	   end if

	   if (dt_cap_yr > 0d0) then
	      s% dt_next = min(s% dt_next, dt_cap_yr*secyer)
	   end if

	   extras_finish_step = keep_going
	end function extras_finish_step


      end module run_star_extras
