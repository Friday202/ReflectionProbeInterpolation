# ReflectionProbeInterpolation

Realistic reflection rendering plays a crucial role in enhancing the visual quality of interactive applications such
as video games and virtual reality experiences. Traditionally, the use of precomputed reflection probes with baked
images has been a popular approach to capture static environments. However, dynamically changing scenes pose
a challenge for real-time reflection updates, demanding efficient techniques that strike a balance between visual
fidelity and computational cost.
In this paper, we propose a method for enhancing quality of traditional reflection capturing method that uses
Delaunay triangulation and a neural network. We introduce a pipeline that could potentially be used to enhance
interpolation images of the captured environment.
To evaluate the performance and visual quality of our method, we conducted a series of experiments in Python
as well as suggest one possible way of incorporating the pipeline in Unreal Engine 5. The results showcase the
potential of using the suggested pipeline as a possible replacement or an enhancement of the traditional reflection
probe linear interpolation.
