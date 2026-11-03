# Mathematical scope


The shared local-law toolkit follows graphical derivatives and basic coderivatives in the sense
of Rockafellar and Wets, *Variational Analysis* (1998). Each supported law supplies its values, selected partial derivatives,
and joint directional action.
General graph composition, topology changes, and degenerate implicit relations require additional
models and qualification conditions outside this toolkit.

For a relation $S$ at a graph point $(p,z)$, the graphical derivative is
$$
DS(p\mid z)[h]=\{d:(h,d)\in T_{\operatorname{gph}S}(p,z)\},
$$
where $T$ is the contingent tangent cone. The response can be empty, unique, or set-valued.
An empty response means that no finite output direction is attainable. It does not mean that
an operation is unsupported.

At equality, the directional action of a maximum is the maximum of its two incoming actions.
For a minimum it is their minimum, and the absolute value at zero responds by $|h|$.
The inputs and any varying bounds enter one joint action. These rules compose through supported
continuous operations and continuously differentiable stages. Independent coordinate actions
cannot in general be added to recover the joint action.

A continuous scalar law with smooth branches and finite limiting slopes $a$ and $b$
from the left and right has the Hadamard response $ah$ for $h<0$ and $bh$ for
$h\geq0$. Its basic coderivative reads limiting normals to the same graph:
$$
D^*S(p\mid z)[\ell]=\{\beta:(\beta,-\ell)\in N^{\mathrm{bas}}_{\operatorname{gph}S}(p,z)\}.
$$
For this scalar law,
$$
D^*S(p\mid z)[\ell]=
\begin{cases}
[\ell a,\ell b], & \ell a\leq\ell b, \\
\{\ell a,\ell b\}, & \ell a>\ell b.
\end{cases}
$$
The order of the seed-weighted slopes matters. The positive part at zero has the interval
$[0,\ell]$ for a nonnegative seed, and the disconnected set $\{0,\ell\}$ for a
negative seed. A convex hull would add covectors in the latter case. No single linear transpose
represents this relation for every seed. For several outputs with one scalar input, contract the
output covector with each incident branch derivative first, then use those two scalar values in
the same rule. This requires continuity and finite smooth incident jets of the complete local
law. An arbitrary directional-action interface does not establish those qualifications.

For moving table knots $x_i$ and ordinates $y_i$, an affine segment with interpolation
fraction $\tau$ and slope $s$ has the joint action
$$
(1-\tau)\dot y_i+\tau\dot y_{i+1}
+s\bigl(\dot x-(1-\tau)\dot x_i-\tau\dot x_{i+1}\bigr).
$$
At knot $i$ the action is $\dot y_i+s_{\pm}(\dot x-\dot x_i)$, with the side
chosen by the relative velocity $\dot x-\dot x_i$. In a scalar reverse query at that knot,
the knot covector is the negative input covector, its ordinate covector is the output seed,
and the other row covectors vanish. Strict positive knot gaps preserve the chart domain.

Regular implicit responses solve the linear state-response equation obtained from the joint
residual action. Parameter kinks can provide nonlinear dependence on the prescribed direction
while the state-response equation remains linear. A vanishing outer first sensitivity can remove
a sampled scalar kink from that equation. The native check treats all tied primitives together,
so one selected branch cannot conceal another primitive's sensitivity. An active state kink with
nonzero sensitivity needs an independent existence and uniqueness argument before a directional
solve can claim a solution response. The toolkit refuses that case. The response matrix is the exact
state Jacobian of the converged residual, even when the primal iteration uses an approximate
Jacobian. Selected linearisations and their transposes retain their branch interpretation on a
persistent regular stratum.

A directional map does not determine a basic coderivative, and primitive coderivatives do not
supply an exact global reverse relation through independent branch choices. Weak-form responses
also depend on the topology of convergence. Spatial differentiation can create a moving jump
whose continuum response needs an interface term. Native finite-quadrature responses and
fixed-topology coordinate actions have the narrower contracts described in
[the native API contracts](api.md).
