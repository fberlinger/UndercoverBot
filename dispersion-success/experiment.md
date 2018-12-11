## Dispersion Experiment

Goal: Learn dispersion

Run fish for 200 generations with populations size 100.
Fish and replica fish run in individual schools.

Important parameters:
- inital_spread: 20
- alpha: 40

- Normalization factor for replica fish: / 100
- Normalization for speed: / 9
- normalization for angular speed: / np.pi

- Use 2 radii as input to optimizer
- Radii taken directly from evolution
- Evolution bounds: -40, 300 (classifier)
- Evolution bounds: -10, 10 (model)
- Evolution sigma: 1 (both model and classifier)


Messed around while it was running around generation 185. Seems to have sigfinicantly messed things up in last 20 generations. Not entirely sure why.

Outcome: before generation 180, seems to learn dispersion behavior. See plots and images from these generations. Dispersion starts to but doesn't completely normalize. Can try aggregation but doesn't really work. Also tried adding some real fish which makes the extreme dispersion more accurate.



# test 2
Try 2 radiii
Important parameters:
- inital_spread: 20
- alpha: 40

- Normalization factor for replica fish: * 0.01
- Normalization for speed: / 9
- normalization for angular speed: / np.pi

- Use 2 radii as input to optimizer
- Evolution bounds: -10, 10 (classifier)
- Evolution bounds: -10, 10 (model)
- Evolution sigma: 0.5 (both model and classifier)

Radius add ten, mutliply by 20/300 to get from evolved weight
Note: bug in classifying radius - want to multiply by inverse.

# Test 3
Fix bug in test 2 - change to mutliply by 300 / 20 for full range of possible radii.
Evolution sigma: 1
Notes: the best model swims in a a curly-Q. Avg. neighbor distance stays small, but keep at max velocity. Radii one ranges mid 20s to mid40s. Radii 2 ranges 60-80, with most about 70. Model scores 0.3 - 0.8, high variance. Class scores 0.3-0.4

Try running to 500 geneations


# Test 4, try test 1 again
Notes: totally successful dispersion. No idea why. Radii almost uniformly negative and approximately 3. Disperses symmetrically but too much. Speed decreases over time. Classifier scores about 0.5. Model scores around 0.4


