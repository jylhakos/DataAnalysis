# Gradient Descent 

Gradient Descent (GD) is an optimization algorithm used to train Large Language Models (LLMs) by iteratively adjusting billions of parameters to minimize loss, typically using variants like AdamW. While traditional gradient descent acts as a "disciplined doer" making locally optimal updates, modern approaches increasingly leverage LLMs themselves as high-level "mentors" or "optimizers" to guide this process, creating a hybrid approach that enhances efficiency and overcomes local optima.

## Training

Process: LLMs are trained by calculating the slope (gradient) of a loss function and taking small steps downhill to minimize error.
Stochastic Gradient Descent (SGD): To handle massive datasets, SGD uses small random batches of data rather than the entire dataset, making training feasible.
Memory Efficiency: Storing gradients for billions of parameters is resource-intensive. Techniques like mixed precision training (float16 instead of float32), parameter offloading, and Gradient Checkpointing are used to manage memory.
Modern Alternatives: New methods like SinkGD and SWAN provide Adam-level performance with the memory footprint of simple SGD, enabling more efficient pretraining. 

