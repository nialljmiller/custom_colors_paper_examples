module run_star_extras
   use star_lib
   use star_def
   use const_def
   implicit none
contains
   subroutine extras_controls(id, ierr)
      integer, intent(in) :: id
      integer, intent(out) :: ierr
      type(star_info), pointer :: s
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      s% extras_check_model => extras_check_model
      s% how_many_extra_history_columns => how_many_extra_history_columns
      s% data_for_extra_history_columns => data_for_extra_history_columns
   end subroutine

   integer function extras_check_model(id)
      integer, intent(in) :: id
      integer :: ierr
      type(star_info), pointer :: s
      extras_check_model = keep_going
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      ! Native log_g_lower_limit uses outermost-cell gravity.
      ! Also guard the photospheric gravity supplied to Colors and MSG.
      if (s% photosphere_logg <= 0.1d0) then
         termination_code_str(t_xtra1) = 'CAP18 photosphere log_g <= 0.1'
         s% termination_code = t_xtra1
         extras_check_model = terminate
      end if
   end function

   integer function how_many_extra_history_columns(id)
      integer, intent(in) :: id
      how_many_extra_history_columns = 4
   end function

   subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
      use colors_def, only: Colors_General_Info
      use colors_lib, only: colors_ptr
      integer, intent(in) :: id, n
      character(len=maxlen_history_column_name) :: names(n)
      real(dp) :: vals(n)
      integer, intent(out) :: ierr
      integer :: k
      type(star_info), pointer :: s
      type(Colors_General_Info), pointer :: settings
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      call colors_ptr(s% colors_handle, settings, ierr)
      if (ierr /= 0) return
      k = s% photosphere_cell_k
      names = [character(len=maxlen_history_column_name) :: &
         'colors_radius_cm', 'colors_log_g', 'metallicity_dex', 'fixed_alpha_MLT']
      vals(1) = s% photosphere_r * Rsun
      vals(2) = s% photosphere_logg
      ! Reproduce the exact atmosphere metallicity used by history.f90.
      vals(3) = minval(settings% lu_meta)
      if (s% X(k) > 0d0 .and. s% Z(k) > 0d0) then
         vals(3) = log10((s% Z(k)/s% X(k))/settings% z_over_x_ref)
         vals(3) = max(minval(settings% lu_meta), min(maxval(settings% lu_meta), vals(3)))
      end if
      vals(4) = s% mixing_length_alpha
   end subroutine
end module run_star_extras
