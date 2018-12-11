# Single Radius and velocity Experiment

Goal: Learn dispersion with radius and velocity to classifier

Run fish for 200 generations with populations size 100.
Fish and replica fish run in individual schools.

Important parameters:
- inital_spread: 20
- alpha: 40

- Normalization factor for replica fish: 0.1
- Normalization for speed: / 9
- normalization for angular speed: / np.pi

- Use 1 radii and speed as input to optimizer
- Radius modified from learned final weight. (add 10, multiply by 300 / 20)
- Evolution bounds: -10, 10 (both model and classifier)
- Evolution sigma: 0.5 (both model and classifier)


- learns to slow down really well
- doesn't learn to disperse


#  Trial 2:
give it 2 radii and velocity. Same number of generations, other parameters

Notes: Learns to slow down super well again, but does not learn to disperse
Class scores low - mostly 0.3
Fish scores variable, but some are high (1)

# Trial 3:
In fish neural net, in output, only do logistic sigmoid for speed output.
Let velocity output stay constant.
Make evolution sigma 0.8

Notes: again learns to slow down super well, but not to disperse. Both radii very large.
Model scores about 0.4-0.5. Class scores 0.3-0.5 (mostly below 0.5)

# Trial 4: Aggregation + Velocity
Initial spread: 100
Alpha: 40

- Normalization factor for replica fish: 0.01
- Normalization for speed: / 9
- normalization for angular speed: / np.pi

- Radius modified from learned final weight. (add 10, multiply by 300 / 20)
- Evolution bounds: -10, 10 (both model and classifier)
- Evolution sigma: 1 (both model and classifier)

Run 300 generations

Actually both aggregated and stopped. Aggregated too close, and only some fish moved.

# Trial 5: Dispersion + Velocity
Initial spread: 20
Alpha: 40

- Normalization factor for replica fish: 0.01
- Normalization for speed: / 9
- normalization for angular speed: / np.pi

- Radius modified from learned final weight. (add 10, multiply by 300 / 20)
- Evolution bounds: -10, 10 (both model and classifier)
- Evolution sigma: 1 (both model and classifier)

Run 300 generations

Notes: didn't aggregate
