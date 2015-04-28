#ifndef STEPPING_H
#define STEPPING_H

void stepping_kick(Particles* const particles, const float Omega_m,
        const float ai, const float af, const float ac);
void stepping_drift(Particles* const particles, const float Omega_m,
        const float ai, const float af, const float ac);

void stepping_set_subtract_lpt(int flag);
void stepping_set_std_da(int flag);
void stepping_set_no_pm(int flag);
void set_nonstepping_initial(const float aout, Particles const * const particles, Snapshot* const snapshot);

void stepping_set_snapshot(const double aout, double a_x, double a_v, Particles const * const particles, Snapshot* const snapshot);

#endif