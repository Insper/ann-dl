Recurrent Neural Networks (RNNs) are powerful for sequence data. Long Short-Term Memory (LSTM) networks are a type of RNN designed to capture long-term dependencies and mitigate issues like vanishing gradients.

**Parameters**:

<div class="grid cards" markdown>
-   

-   (Simplified to hidden size=1 for clarity):

    - Inputs:

        \( x_t = [0.5] \),
        
        \( h_{t-1} = [0.1] \),
        
        \( C_{t-1} = [0.2] \)

    - Weights:

        \( W_f = [[0.5, 0.5]] \),

        \( W_i = [[0.4, 0.4]] \),
        
        \( W_C = [[0.3, 0.3]] \),
        
        \( W_o = [[0.2, 0.2]] \)

    - Biases: \( b_f = b_i = b_C = b_o = [0.0] \)
</div>

**Forward Pass**:

<div class="grid cards" markdown>

-   - Concatenate: \( \text{concat} = [h_{t-1}, x_t] \)
    - Forget gate: \( f_t = \sigma(W_f \cdot \text{concat} + b_f) \)
    - Input gate: \( i_t = \sigma(W_i \cdot \text{concat} + b_i) \)
    - Cell candidate: \( \tilde{C}_t = \tanh(W_C \cdot \text{concat} + b_C) \)
    - Cell state: \( C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t \)
    - Output gate: \( o_t = \sigma(W_o \cdot \text{concat} + b_o) \)
    - Hidden state: \( h_t = o_t \cdot \tanh(C_t) \)

-   - concat = [0.1, 0.5]
    - \( f_t = \sigma(0.3) \approx 0.5744 \)
    - \( i_t = \sigma(0.24) \approx 0.5597 \)
    - \( \tilde{C}_t = \tanh(0.18) \approx 0.1785 \)
    - \( C_t \approx 0.5744 \cdot 0.2 + 0.5597 \cdot 0.1785 \approx 0.2146 \)
    - \( o_t = \sigma(0.12) \approx 0.5300 \)
    - \( h_t \approx 0.5300 \cdot \tanh(0.2146) \approx 0.1120 \)

</div>

**Backward Pass**:

<div class="grid cards" markdown>

-   Gradients are computed via chain rule:

    - \( dC_t = dh_t \cdot o_t \cdot (1 - \tanh^2(C_t)) + dC_{next} \) (dC_next from future timestep)
    - \( do_t = dh_t \cdot \tanh(C_t) \cdot \sigma'(o_t) \)
    - \( d\tilde{C}_t = dC_t \cdot i_t \cdot (1 - \tilde{C}_t^2) \)
    - \( di_t = dC_t \cdot \tilde{C}_t \cdot \sigma'(i_t) \)
    - \( df_t = dC_t \cdot C_{t-1} \cdot \sigma'(f_t) \)
    - \( dC_{prev} = dC_t \cdot f_t \)
    - Then, backpropagate to concat: \( d\text{concat} = W_o^T \cdot do_t + W_C^T \cdot d\tilde{C}_t + W_i^T \cdot di_t + W_f^T \cdot df_t \)
    - Split \( d\text{concat} \) into \( dh_{prev} \) and \( dx_t \)
    - Parameter gradients: \( dW_f = df_t \cdot \text{concat}^T \), \( db_f = df_t \), and similarly for others.

-   (Assume upstream: \( dh_t = [0.1] \), \( dC_t = [0.05] \) from next timestep):

    - \( dC_t \approx 0.1 \cdot 0.5300 \cdot (1 - \tanh^2(0.2146)) + 0.05 \approx 0.1028 + 0.05 = 0.1528 \) (detailed steps in code output)
    - Resulting gradients match the executed values below (e.g., \( dx_t \approx [0.0216] \), etc.).

</div>

```pyodide install="pandas,scikit-learn" exec="on" html="1"
--8<-- "docs/classes/deep-learning/lstm.py"
```

!!! info "Notes"

    - This is a single-timestep LSTM with hidden size 1 for simplicity. In practice, LSTMs process sequences (multiple timesteps) and have larger hidden sizes; backpropagation through time (BPTT) unrolls the network over timesteps.
    - The code uses NumPy; for real models, use PyTorch or TensorFlow for automatic differentiation and batching.
    - Outputs are approximate due to floating-point precision but match the manual calculations.
    - If you need a multi-timestep example, sequence processing, or integration into a full RNN, let me know!