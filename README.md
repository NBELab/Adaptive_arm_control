# Adaptive Control of a Wheelchair Mounted Robotic Arm with Neuromorphically Integrated Velocity Readings and Online-Learning

### Michael Ehrlich, Yuval Zaidel, Patrice L. Weiss, Arie Melamed Yekel, Naomi Gefen, Lazar Supic, Elishai Ezra Tsur

This repository represent the implementation of our work, submitted for consideration in the Frontier of Neuroscience.

### Project abstract

Wheelchair mounted robotic arms support people with upper extremity disabilities with various activities of daily living. However, the associated cost and the energy efficiency of responsive and adaptive assistive robotic arms contribute to the fact that such systems are in limited use. Neuromorphic spiking neural networks can be used for a real-time machine learning-driven control of robots supporting an adaptive and responsive behavior in stochastic environmental conditions. In this work, we demonstrate a neuromorphic adaptive control of a wheelchair-mounted robotic arm deployed on Intel’s Loihi chip. Our algorithm design uses neuromorphically represented and integrated velocity readings to derive the arm’s current state, providing the controller with motion guidance and adaptive signals, and allowing it to account for kinematic changes in real-time. We pilot tested the device with an able-bodied participant to evaluate its accuracy while performing activities of daily living (ADL)-related trajectories and to demonstrate the capacity of the controller to compensate for unexpected inertia-generating payloads using online learning in real-life application. Videotaped recordings of two ADL tasks performed by the robot were viewed by 10 caregivers; data summarizing their feedback on the user experience and the potential benefit of the system is reported.

### Prerequisites
Python (anaconda distribution). availible in: https://www.anaconda.com/products/distribution

MuJoCo simulation: https://mujoco.org/

### Execution
Execute the Jupyter file "Arm_control.ipynb"" to run the robotic simulation with and without external force vectors.


