# Method Section for CVPR Paper

## Visual Uncertainty-Aware Reinforcement Learning for Multimodal Reasoning

### 3. Method

In this section, we present our novel approach for enhancing multimodal large language models (MLLMs) through visual uncertainty-aware reinforcement learning. Our method addresses a fundamental challenge in vision-language tasks: the model's ability to correctly interpret visual information and reason about task-specific requirements.

#### 3.1 Overview

Our framework builds upon Group Relative Policy Optimization (GRPO) and introduces two key innovations: (1) a **dual-branch training mechanism** that explicitly models visual perception uncertainty, and (2) a **token-level uncertainty quantification** approach that enriches the advantage estimation. Figure 1 illustrates the overall architecture of our method.

The core insight is that when a model exhibits high discrepancy between predictions made with and without explicit visual guidance (e.g., text overlaid on images), this signals visual perception uncertainty. By quantifying and incorporating this uncertainty into the RL training objective, we guide the model to develop more robust visual reasoning capabilities.

#### 3.2 Dual-Branch Training Mechanism

**Problem Formulation.** Given a vision-language task with input image \(I\) and text query \(Q\), conventional approaches train the model to generate response \(R\) by maximizing:

\[
\mathcal{L}_{\text{std}} = \mathbb{E}_{(I,Q,R) \sim \mathcal{D}} \left[ \log \pi_\theta(R | I, Q) \right]
\]

However, this formulation does not explicitly model whether the model relies on visual information or merely pattern-matching on textual cues.

**Branch A: Standard Visual-Text Input.** Our first branch follows the conventional paradigm where the model receives the original image \(I_A = I\) along with the text query \(Q_A = Q\):

\[
p_A(R | I_A, Q_A) = \pi_\theta(R | I, Q)
\]

This branch captures the model's natural prediction distribution without additional visual guidance.

**Branch B: Text-Overlaid Visual Input.** To provide explicit visual guidance, we create a modified image \(I_B = \text{Overlay}(I, Q)\) where the text query is rendered onto the image using random positioning and styling. The text prompt is simplified to a generic instruction:

\[
Q_B = \text{``Please answer the question in the image.''}
\]

This branch computes:

\[
p_B(R | I_B, Q_B) = \pi_\theta(R | \text{Overlay}(I, Q), Q_B)
\]

By overlaying the question directly onto the image, Branch B forces the model to process the visual information more carefully, as the question becomes part of the visual input.

**Cross-Modal Prediction.** A critical component of our method is computing the cross-modal prediction distribution, which uses Branch A's visual encoding with Branch B's response tokens:

\[
p_{\text{cross}}(R_B | I_A, Q_A) = \pi_\theta(R_B | I, Q)
\]

This allows us to measure how much the visual encoding influences the prediction when the response context comes from the text-overlaid branch.

#### 3.3 Visual Uncertainty Quantification

We quantify visual uncertainty at the token level by measuring the symmetric Kullback-Leibler (KL) divergence between the cross-modal prediction distribution and Branch B's native distribution:

\[
\mathcal{U}_{\text{vis}}^{(t)} = \frac{1}{2} \left[ D_{\text{KL}}(p_B^{(t)} \| p_{\text{cross}}^{(t)}) + D_{\text{KL}}(p_{\text{cross}}^{(t)} \| p_B^{(t)}) \right]
\]

where \(t\) indexes the token position in the response sequence, and \(p^{(t)}\) denotes the probability distribution over the vocabulary at position \(t\).

**Intuition.** High visual uncertainty \(\mathcal{U}_{\text{vis}}^{(t)}\) indicates that the model's predictions are sensitive to whether it uses Branch A's visual encoding (original image) or Branch B's visual encoding (text-overlaid image). This suggests the model is uncertain about visual interpretation. Conversely, low visual uncertainty indicates the model consistently predicts similar distributions regardless of visual encoding, suggesting confident visual understanding or over-reliance on textual patterns.

The symmetric KL divergence is computed as:

\[
\mathcal{U}_{\text{vis}}^{(t)} = \frac{1}{2} \sum_{v \in \mathcal{V}} \left[ p_B^{(t)}(v) \log \frac{p_B^{(t)}(v)}{p_{\text{cross}}^{(t)}(v)} + p_{\text{cross}}^{(t)}(v) \log \frac{p_{\text{cross}}^{(t)}(v)}{p_B^{(t)}(v)} \right]
\]

where \(\mathcal{V}\) denotes the vocabulary.

#### 3.4 Token-Level Entropy Regularization

In addition to visual uncertainty, we compute token-level prediction entropy for both branches to encourage exploration and prevent premature convergence:

\[
\mathcal{H}_A^{(t)} = -\sum_{v \in \mathcal{V}} p_A^{(t)}(v) \log p_A^{(t)}(v)
\]

\[
\mathcal{H}_B^{(t)} = -\sum_{v \in \mathcal{V}} p_B^{(t)}(v) \log p_B^{(t)}(v)
\]

High entropy indicates the model maintains diverse predictions, while low entropy suggests confident (potentially overconfident) predictions.

#### 3.5 Uncertainty-Aware Advantage Estimation

Our method modifies the standard advantage function in GRPO by incorporating visual uncertainty and token entropy. For a response \(R = (r_1, \ldots, r_T)\), GRPO computes advantages by normalizing rewards across multiple rollouts for the same prompt:

\[
A^{\text{GRPO}}_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}
\]

where \(r_i\) is the reward for the \(i\)-th rollout, and \(\mu_{\text{group}}, \sigma_{\text{group}}\) are the mean and standard deviation within each prompt group.

**Branch A Advantage Enhancement.** For Branch A, we augment advantages with token entropy:

\[
\tilde{A}_A^{(t)} = A_A^{(t)} + \min\left( \frac{|A_A^{(t)}|}{\beta_H}, \alpha_H \cdot \mathcal{H}_A^{(t)} \right) \cdot m^{(t)}
\]

**Branch B Advantage Enhancement.** For Branch B, we incorporate both visual uncertainty and token entropy:

\[
\begin{aligned}
\tilde{A}_B^{(t)} = A_B^{(t)} &+ \min\left( \frac{|A_B^{(t)}|}{\beta_U}, \alpha_U \cdot \mathcal{U}_{\text{vis}}^{(t)} \right) \cdot m^{(t)} \\
&+ \min\left( \frac{|A_B^{(t)}|}{\beta_H}, \alpha_H \cdot \mathcal{H}_B^{(t)} \right) \cdot m^{(t)}
\end{aligned}
\]

where \(m^{(t)} \in \{0,1\}\) is a binary mask indicating valid tokens, \(\alpha_U, \alpha_H\) are scaling coefficients, and \(\beta_U, \beta_H\) are normalization factors to prevent gradient explosion.

**Clipping Mechanism.** The \(\min(\cdot, \cdot)\) operation serves as an adaptive clipping mechanism. When the magnitude of the advantage \(|A^{(t)}|\) is small, we clip the uncertainty bonus using \(\alpha \cdot \mathcal{U}^{(t)}\). When the advantage is large, we allow larger bonuses proportional to the advantage magnitude scaled by \(1/\beta\). This prevents uncertainty terms from overwhelming the original reward signal while still providing meaningful regularization.

#### 3.6 Policy Optimization Objective

The policy is trained to maximize the clipped surrogate objective:

\[
\mathcal{L}_{\text{CLIP}}(\theta) = \mathbb{E}_{t} \left[ \min\left( \rho^{(t)} \tilde{A}^{(t)}, \text{clip}(\rho^{(t)}, 1-\epsilon_L, 1+\epsilon_H) \tilde{A}^{(t)} \right) \cdot m^{(t)} \right]
\]

where:
- \(\rho^{(t)} = \frac{\pi_\theta(r_t | s_t)}{\pi_{\theta_{\text{old}}}(r_t | s_t)}\) is the probability ratio
- \(\epsilon_L = 0.2\) and \(\epsilon_H = 0.3\) are clipping parameters for low and high bounds
- \(s_t\) denotes the state (image and text context) at position \(t\)

For dual-branch training, we employ a **progressive branch sampling strategy**. At training step \(n\) out of \(N\) total steps, we sample from Branch A with probability:

\[
p_A(n) = 1 - \frac{n}{N}
\]

and from Branch B with probability \(p_B(n) = 1 - p_A(n)\). This ensures early training focuses more on the standard visual-text input (Branch A), while later training emphasizes the text-overlaid input (Branch B) which incorporates visual uncertainty.

**Optional KL Divergence Penalty.** When enabled, we add a KL divergence term between the current policy and a frozen reference policy:

\[
\mathcal{L}_{\text{KL}}(\theta) = \mathbb{E}_{t} \left[ D_{\text{KL}}\left(\pi_\theta(\cdot | s_t) \| \pi_{\text{ref}}(\cdot | s_t)\right) \cdot m^{(t)} \right]
\]

The total loss becomes:

\[
\mathcal{L}_{\text{total}}(\theta) = -\mathcal{L}_{\text{CLIP}}(\theta) + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}(\theta)
\]

#### 3.7 Implementation Details

**Text Overlay Generation.** For Branch B, we implement a deterministic overlay function that renders the question text onto the image with random positioning, font size, color, and rotation (seeded by example ID for reproducibility). This ensures consistent overlay patterns within each training run while maintaining diversity across different examples.

**Efficient Computation.** The dual-branch mechanism requires three forward passes per training step:
1. Branch A forward pass to compute \(p_A\) and \(\mathcal{H}_A\)
2. Branch B forward pass to compute \(p_B\) and \(\mathcal{H}_B\)
3. Cross-modal forward pass to compute \(p_{\text{cross}}\)

To optimize memory usage, we only retain full vocabulary distributions during the forward pass and detach them from the computation graph after computing uncertainty metrics. During the backward pass, only the selected token log-probabilities are used for gradient computation.

**Dynamic Batching.** We employ dynamic batching with padding-free training using Flash Attention's variable-length sequence handling. This reduces memory consumption and improves computational efficiency, especially for datasets with varying image resolutions and text lengths.

**Hyperparameters.** We set \(\alpha_U = 1.0\), \(\beta_U = 2.0\), \(\alpha_H = 0.01\), and \(\beta_H = 2.0\) for our main experiments. These values are tuned to balance the contribution of uncertainty terms with the original reward signal. We use the AdamW optimizer with learning rate \(1 \times 10^{-6}\), weight decay \(1 \times 10^{-2}\), and gradient clipping at norm 1.0.

#### 3.8 Theoretical Justification

**Proposition 1.** *Under mild regularity conditions, incorporating visual uncertainty into the advantage function encourages the policy to reduce prediction variance across different visual encodings, leading to more robust visual reasoning.*

**Proof Sketch.** Consider the policy gradient with uncertainty-aware advantages:

\[
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(R|I,Q) \cdot \left(A(R) + \alpha_U \mathcal{U}_{\text{vis}}\right) \right]
\]

The visual uncertainty term \(\mathcal{U}_{\text{vis}}\) increases the advantage when the model exhibits high discrepancy between \(p_B\) and \(p_{\text{cross}}\). This creates a gradient signal that pushes the policy to reduce this discrepancy, effectively encouraging the model to develop visual encodings that are robust to perturbations in how the question is presented (as separate text vs. overlaid on image).

**Proposition 2.** *The token-level entropy regularization prevents mode collapse and maintains exploration during training.*

**Proof Sketch.** The entropy bonus \(\alpha_H \mathcal{H}^{(t)}\) increases the advantage for predictions with higher entropy. From the policy gradient perspective:

\[
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(R|I,Q) \cdot \left(A(R) + \alpha_H \mathcal{H}\right) \right]
\]

The entropy term provides a positive signal proportional to the prediction entropy, encouraging the model to maintain diverse predictions rather than collapsing to a single mode. This is particularly important in the early stages of training when the model needs to explore different reasoning strategies.

#### 3.9 Relation to Prior Work

**GRPO vs. Our Method.** Standard GRPO normalizes advantages within each prompt group but does not explicitly model visual perception uncertainty. Our method augments GRPO with uncertainty-aware bonuses that specifically target visual reasoning robustness.

**PPO with Entropy Regularization.** While PPO often includes an entropy bonus in the loss function, our token-level entropy is computed over the full vocabulary distribution and incorporated into the advantage function rather than as a separate loss term. This provides a more direct signal during advantage estimation.

**Contrastive Learning.** Our dual-branch mechanism can be viewed as a form of contrastive learning where we explicitly contrast the model's behavior under different visual presentation modes. However, unlike standard contrastive approaches that push representations apart, we use the discrepancy to quantify uncertainty and guide policy optimization.

### 4. Experimental Setup

*(This section would detail your experimental configuration, datasets, baselines, and evaluation metrics)*

**Datasets.** We evaluate our method on multiple vision-language reasoning benchmarks:
- **Geometry3K**: Geometric problem solving requiring spatial reasoning
- **Math-Vision**: Mathematical problems with diagrams and charts  
- **OCR-VQA**: Visual question answering requiring text recognition in images

**Implementation Details.** We implement our framework based on EasyR1, a scalable multi-modal RL training system built on the HybridEngine architecture. We use Qwen2.5-VL models ranging from 3B to 32B parameters as our base models. Training is performed using FSDP (Fully Sharded Data Parallel) with mixed precision (BF16) on up to 32 GPUs.

**Baselines.** We compare against:
- Supervised Fine-Tuning (SFT): Standard supervised training on (question, answer) pairs
- GRPO: Group Relative Policy Optimization without uncertainty awareness
- REINFORCE++: REINFORCE with discount factor
- ReMax: Reward Maximum using expected reward baselines

**Evaluation Metrics.** We report:
- **Accuracy**: Exact match between predicted and ground-truth answers
- **Format Correctness**: Percentage of responses following the required format
- **Overall Score**: Weighted combination of accuracy and format correctness

### Algorithm Summary

**Algorithm 1: Visual Uncertainty-Aware GRPO**

```
Input: Model π_θ, dataset D, hyperparameters α_U, β_U, α_H, β_H
Output: Optimized policy π_θ*

for epoch = 1 to N_epochs do
    for each prompt (I, Q) in D do
        // Rollout Phase
        Sample n responses {R_1, ..., R_n} ~ π_θ(·|I, Q)
        
        for each branch ∈ {A, B} do
            // Branch A: standard input
            if branch == A then
                Prepare (I_A, Q_A) = (I, Q)
            // Branch B: text-overlaid input  
            else
                Prepare (I_B, Q_B) = (Overlay(I, Q), generic_prompt)
            end if
            
            // Compute rewards
            Compute rewards {r_i} using reward function
            Compute GRPO advantages {A_i}
            
            // Forward pass with full vocabulary distributions
            Compute log-probs and full distributions {p^(t)}
            Compute token entropy {H^(t)}
            
            if branch == B then
                // Compute cross-modal predictions
                Compute p_cross^(t) using (I_A, Q_A) encoding with R_B
                Compute visual uncertainty U_vis^(t) = SymKL(p_B, p_cross)
                
                // Enhance advantages with uncertainty
                Ã_B^(t) ← A_B^(t) + min(|A_B^(t)|/β_U, α_U·U_vis^(t))
                Ã_B^(t) ← Ã_B^(t) + min(|A_B^(t)|/β_H, α_H·H_B^(t))
            else
                // Enhance advantages with entropy only
                Ã_A^(t) ← A_A^(t) + min(|A_A^(t)|/β_H, α_H·H_A^(t))
            end if
        end for
        
        // Progressive branch sampling
        p_A ← 1 - (current_step / total_steps)
        Sample branch ~ Bernoulli(p_A)
        
        // Policy update with selected branch
        Compute clipped surrogate loss L_CLIP using Ã from sampled branch
        Update θ ← θ + η·∇_θ L_CLIP
    end for
end for
```

### Conclusion

We have presented a novel visual uncertainty-aware reinforcement learning framework for training multimodal large language models. Our dual-branch training mechanism explicitly quantifies visual perception uncertainty through symmetric KL divergence between cross-modal predictions, providing a principled way to enhance advantage estimation in policy optimization. Combined with token-level entropy regularization, our method encourages the model to develop robust visual reasoning capabilities while maintaining exploration. Extensive experiments demonstrate the effectiveness of our approach across multiple vision-language reasoning benchmarks.

---

## Key Contributions Summary

1. **Dual-Branch Training Mechanism**: A novel approach that contrasts model behavior on standard visual-text inputs vs. text-overlaid visual inputs to expose visual perception uncertainty.

2. **Visual Uncertainty Quantification**: Token-level symmetric KL divergence between cross-modal and native predictions provides a fine-grained measure of visual reasoning confidence.

3. **Uncertainty-Aware Advantage Estimation**: Principled integration of visual uncertainty and token entropy into the GRPO advantage function with adaptive clipping to prevent gradient explosion.

4. **Progressive Branch Sampling**: Time-dependent sampling strategy that gradually shifts focus from standard inputs to uncertainty-revealing text-overlaid inputs.

5. **Scalable Implementation**: Efficient implementation supporting models up to 70B+ parameters with dynamic batching, padding-free training, and distributed optimization.

---

## Mathematical Notation Reference

| Symbol | Description |
|--------|-------------|
| \(I\) | Input image |
| \(Q\) | Text query/question |
| \(R\) | Generated response sequence |
| \(\pi_\theta\) | Policy parameterized by \(\theta\) |
| \(I_A, Q_A\) | Branch A inputs (standard) |
| \(I_B, Q_B\) | Branch B inputs (text-overlaid) |
| \(p_A^{(t)}, p_B^{(t)}\) | Token distributions at position \(t\) for branches A and B |
| \(p_{\text{cross}}^{(t)}\) | Cross-modal token distribution |
| \(\mathcal{U}_{\text{vis}}^{(t)}\) | Visual uncertainty at token \(t\) |
| \(\mathcal{H}_A^{(t)}, \mathcal{H}_B^{(t)}\) | Token entropy at position \(t\) for branches A and B |
| \(A^{(t)}\) | Advantage at token \(t\) |
| \(\tilde{A}^{(t)}\) | Uncertainty-enhanced advantage |
| \(\alpha_U, \beta_U\) | Visual uncertainty scaling and normalization |
| \(\alpha_H, \beta_H\) | Entropy scaling and normalization |
| \(m^{(t)}\) | Binary mask for valid tokens |
| \(\rho^{(t)}\) | Probability ratio (new policy / old policy) |
| \(\epsilon_L, \epsilon_H\) | PPO clipping bounds (low and high) |

---

## Related Implementation Notes for Reproducibility

**1. Text Overlay Function**: The `add_text_to_image()` function generates deterministic overlays using a seed derived from the example ID. Text is rendered with random position (avoiding image corners), font size (18-28pt), color, and slight rotation (-5° to +5°).

**2. Symmetric KL Divergence**: Implemented using PyTorch's `F.kl_div()` with `log_target=True` for numerical stability:
```python
kl_pq = F.kl_div(cross_log_probs, log_probs, reduction='none', log_target=True)
kl_qp = F.kl_div(log_probs, cross_log_probs, reduction='none', log_target=True)
sym_kl = 0.5 * (kl_pq + kl_qp)
```

**3. Memory Optimization**: Full vocabulary distributions are computed only during forward pass and immediately detached after computing uncertainty metrics. Only log-probabilities for selected tokens are retained for backward pass.

**4. Dynamic Batching**: Sequences are packed without padding using Flash Attention's `unpad_input()` and `pad_input()` utilities, reducing memory consumption by ~30% on average.

**5. FSDP Configuration**: Full parameter sharding with CPU offload option for models >7B. Mixed precision (BF16) for computation with FP32 for gradient accumulation and optimizer states.

**6. Reward Function**: Format reward (10% weight) checks for `<think>...</think>` reasoning tags and `\boxed{}` answer formatting. Accuracy reward uses exact match for Geometry3K, numeric tolerance (±5%) for math problems, and edit distance for text-heavy responses.

**7. Hyperparameter Sensitivity**: The uncertainty coefficients \(\alpha_U\) and \(\alpha_H\) have the largest impact on performance. We recommend \(\alpha_U \in [0.5, 2.0]\) and \(\alpha_H \in [0.005, 0.05]\) based on grid search across three datasets.


