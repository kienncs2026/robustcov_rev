# Logic map: manuscript statement -> code

| Manuscript item | Where | What it does |
|---|---|---|
| Eq. (6) UPA steering, d = λ/2, sign e^{-j...} | `geometry.upa_steering` | unchanged |
| Eq. (15) CV prediction, Q continuous white-noise accel. | `sensing.propagate_targets`, `process_covariance` | unchanged |
| Eq. (16)-(18) measurement and Jacobian (Doppler sign −) | `sensing.measurement_function/jacobian` | unchanged; FD check in `tests/test_models.py` |
| Prop. 1, Eq. (25)-(26) | `theory.psi_spectrum`, `gamma_star`, `pmin_point` | scalar PCRB, bisection for γ*, P_min |
| Prop. 2, Eq. (37) | `covariance.solve_covariance_slot` (`pcrb_formulation="linear"`), `theory.illumination_rows` | one linear inequality per enforced state replaces the 6×6 Schur LMI |
| Prop. 3, Eq. (38)-(39) | `covariance.build_enforcement_sets` (`pcrb_enforcement="scenario_certified"`), `theory.scenario_sample_size` | N_s from (d, ρ, β); untightened ε; sampled range |
| Lemma 4, Eq. (35) | `theory.nine_point_covering_radius`, `theory.whitened_lipschitz`, `posthoc.lipschitz_table` | covering radius; empirical L_w and margin check |
| Algorithm 2 | `refinement.refine_sensing` | validation draws → add largest-deficit violators → re-solve; final test is the independent audit |
| Eq. (41)-(42) terminal reconstruction | `dynamics.enforce_terminal_with_last_two` | unchanged (last two slot accelerations) |
| Eq. (43) SLSQP merit | `trajectory_opt.update_trajectory/_trajectory_penalty` | unchanged; `trajectory_method="random_search"` adds a derivative-free alternative |
| Restoration phase F0 | `feasibility.restore_feasible_trajectory` | now uses the same enforcement sets as the covariance problem |
| Phase I (slack) | `ao_solver._phase1` (`phase1_enabled`) | minimizes the slack value Φ of the covariance problem over trajectory candidates |
| Prop. 4, Eq. (44) | `theory.rf_lower_bound_slot`, `ao_solver._energy_lower_bound` (`candidate_screening`) | exact screening before a candidate re-solve |
| Prop. 5, Eq. (45) | `theory.joint_illumination_bound`, `joint_bound_report`; check in `ao_solver.run_simulation` (`joint_bound_check`) | primal SDP + rescaled dual certificate; slot 1 checked before restoration; failure code `INFEASIBLE_CERTIFIED_BOUNDARY` |
| Cor. 1 | `paper_outputs` → `F5_mobility_decomposition.csv` | propulsion / RF split and RF-share bound |
| Prop. 6 stopping rule | `ao_solver.run_simulation` (`stopping_rule="accepted_gain"`) | stop when the last accepted gain ≤ outer_tolerance; reuse candidate covariances |
| Eq. (47) cost | `summary.csv` columns `time_*`, `covariance_passes` | per-phase wall time |
| Lemma 2, Eq. (24) | `audit.audit_solution` → `lemma2_bcrb_check.csv` | BCRB from the Monte Carlo draws vs mean conditional PCRB vs (1−ρ)ε+ρψ0 |
| Lemma 3 | `theory.closed_vs_open_loop` → `lemma3_closed_vs_open.csv` | full recursion vs open loop at the mean path |
| R1.4 mismatch | `audit._rician_outage`, `audit._coordinated_turn_states`; `posthoc.mismatch_sweep` | Rician K-factor channels; CT truth with CV predictor |
| R2.5 propulsion-only baseline | `baselines.propulsion_only_trajectory` (`scheme.trajectory_objective`) | min propulsion energy under kinematics only, then one covariance solve |
| R3.4 Gaussian randomization | `covariance._gaussian_randomization_bank` (`rank_recovery_mode`) | w ~ CN(0, W_k), powers re-optimized |
