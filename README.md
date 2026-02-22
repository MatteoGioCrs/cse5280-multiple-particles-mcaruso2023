1. Directional Dependence and Symmetry

In the Isotropic model, the social force field is perfectly **symmetrical**. This means the repulsion is based purely on distance; if a particle is 0.5 units away, it pushes back with the same intensity regardless of whether it is in front, behind, or to the side.

In the Anisotropic model, we introduce **directional dependence** via the  parameter. This breaks the symmetry. By weighting the dot product between the velocity vector () and the relative position of the neighbor, the repulsion field becomes "tear-drop" shaped—stronger in the direction of motion and weaker (or zero) behind the particle.

2. Oscillations and Deadlocks

The Anisotropic model significantly reduces deadlocks. In an Isotropic system, two particles heading toward each other create a "force standoff" where they push each other back with equal strength, causing them to jitter (oscillate) or stop entirely (deadlock). In the Anisotropic system, the "broken symmetry" means a particle doesn't push back against someone behind it. This allows the "pressure" from the back of the crowd to actually push the front particles forward toward the goal, rather than creating a counter-force that jams the whole group.

3. Behavior at Bottlenecks and Corridors

* Isotropic Model: In narrow corridors, particles tend to form a "clog." Because they maintain a strict circular personal space, they fight to keep distance from everyone simultaneously. This leads to the "arched" formation at the door where no one can move because everyone is pushing against everyone else.
* Anisotropic Model: Near bottlenecks, this model encourages "lane formation" or "tailgating." Particles allow others to get closer to their backs, which streamlines the group. They act more like a fluid flowing through a pipe rather than a pile of solid spheres trying to squeeze through a hole.

4. Trade-offs of Velocity-Dependent Interactions

While the anisotropic model is more "human-like" and efficient, it introduces specific trade-offs:

Computational Complexity: You must now track and update velocity vectors for every particle, which requires more memory and processing power per frame than a simple distance check.
* Stability Sensitivity: If the  value is too high, particles might ignore obstacles behind them so completely that they become unstable if they are forced to move backward (like being pushed by a wall), potentially leading to unrealistic overlaps or "teleporting" artifacts in the simulation.
* Parameter Tuning: It adds more variables (like ) that must be carefully tuned to prevent the simulation from looking either too rigid (like marbles) or too "ghost-like" (where they phase through each other's backs).

### Isotropic Model
<img width="599" height="594" alt="Isotropic Model 2" src="https://github.com/user-attachments/assets/798d683d-69aa-48a7-9a4f-ac4e0a1e458b" />
### Anisotropic Model
<img width="599" height="594" alt="Anisotropic Model 2" src="https://github.com/user-attachments/assets/bf0c5415-2702-45ac-ba0d-58f224ae527b" />
