import osqp
import numpy as np
import scipy.sparse as sp

"""
    This file contains the implementation of a QP filter for control barrier functions (CBDs). The QP filter is initialised with a CBF object as defined in load_CBF.py.

    To use the filter, state and input control are passed to the filter, which returns a filtered control that satisfies the CBF consraints.

    First there is a function that checks if it will violate the CBF constraints, and if it does, it will solve a QP to find a new control that satisfies the constraints. If it does not violate the constraints, it will return the original control.
"""

class QPFilter:
    def __init__(self, cbf, model, alpha=1.0, u_min=None, u_max=None, slack_penalty=1e5):
        self.cbf = cbf
        self.model = model
        self.alpha = alpha

        # Physical actuator bounds (unicycle: [v, omega])
        self.u_min = np.asarray(u_min) if u_min is not None else np.array([-np.inf, -np.inf])
        self.u_max = np.asarray(u_max) if u_max is not None else np.array([np.inf, np.inf])
        
        self.n_controls = len(self.u_min)
        self.slack_penalty = slack_penalty
        self.use_slack = slack_penalty is not None
        
        # Initialize the OSQP solver once to avoid runtime memory allocations
        self.prob = osqp.OSQP()
        self._init_solver()

    def _build_A(self, Lg_h):
        n_vars = self.n_controls + (1 if self.use_slack else 0)
        rows, cols, data = [], [], []

        for i in range(self.n_controls):
            rows.append(i)
            cols.append(i)
            data.append(1.0)

        next_row = self.n_controls

        if self.use_slack:
            rows.append(next_row)
            cols.append(n_vars - 1)
            data.append(1.0)
            next_row += 1

        for i in range(self.n_controls):
            rows.append(next_row)
            cols.append(i)
            data.append(float(Lg_h[i]))
        if self.use_slack:
            rows.append(next_row)
            cols.append(n_vars - 1)
            data.append(1.0)

        n_constraints = next_row + 1
        return sp.coo_matrix((data, (rows, cols)), shape=(n_constraints, n_vars)).tocsc()

    def _init_solver(self):
        n_vars = self.n_controls + (1 if self.use_slack else 0)

        diag_P = np.ones(n_vars)
        if self.use_slack:
            diag_P[-1] = self.slack_penalty * 2.0
        P = sp.diags(diag_P, format="csc")
        
        placeholder_Lg_h = np.ones(self.n_controls)
        A = self._build_A(placeholder_Lg_h)

        n_constraints = A.shape[0]

        l = np.full(n_constraints, -np.inf)
        u = np.full(n_constraints, np.inf)

        self.prob.setup(P, np.zeros(n_vars), A, l, u, verbose=False, warm_start=True)

    def filter_control(self, state, u_nom):
        u_nom = np.clip(np.asarray(u_nom, dtype=float), self.u_min, self.u_max)

        h = self.cbf.get_cbf_value(state)
        grad_h = self.cbf.get_cbf_gradient(state)

        if hasattr(self.model, "drift_dynamics"):
            f_x = self.model.drift_dynamics(state)
            Lf_h = grad_h @ f_x
        else:
            Lf_h = 0.0

        g = self.model.control_matrix(state)
        Lg_h = grad_h @ g

        alpha_h = self.alpha * (h ** 3)
        
        constraint = Lf_h + Lg_h @ u_nom + alpha_h
            
        if constraint >= 0:
            return u_nom 

        return self._solve_qp(Lg_h, Lf_h, alpha_h, u_nom)

    def _solve_qp(self, Lg_h, Lf_h, alpha_h, u_nom):
        if self.use_slack:
            q = np.append(-u_nom, 0.0)
        else:
            q = -u_nom
        
        A = self._build_A(Lg_h)

        slack_l = [0.0] if self.use_slack else []
        slack_u = [np.inf] if self.use_slack else []

        l = np.concatenate([self.u_min, slack_l, [-alpha_h - Lf_h]])
        u_bound = np.concatenate([self.u_max, slack_u, [np.inf]])

        self.prob.update(q=q, l=l, u=u_bound, Ax=A.data)
        res = self.prob.solve()

        if res.info.status != 'solved':
            return np.zeros(self.n_controls)
        
        return res.x[:self.n_controls]

