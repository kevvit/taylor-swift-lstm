from collections import defaultdict
from copy import deepcopy

import numpy as np
from op import sigmoid, softmax, tanh

class LSTMClassifier:
    def __init__(
            self,
            embed_size: int,
            hidden_size: int,
            vocab_size: int,
            n_cells: int = 1,
            album_count: int = 11,
            album_embedding_size: int = 64
    ) -> None:

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_cells = n_cells
        self.album_count = album_count
        self.album_embedding_size = album_embedding_size
        self.layers = dict()

        self.layers["embedding"] = np.empty((vocab_size, embed_size))
        self.layers["album_embedding"] = np.empty((album_count, album_embedding_size))

        for cell_index in range(n_cells):

            for layer_name in ["f", "o", "c"]:

                linp_sz = hidden_size + (
                    embed_size + album_embedding_size if cell_index == 0 else hidden_size
                )

                self.layers[f"W{layer_name}_{cell_index}"] = np.empty(
                    (linp_sz, hidden_size)
                )
                self.layers[f"b{layer_name}_{cell_index}"] = np.empty(
                    (hidden_size)
                )

        self.layers["W_head"] = np.empty((hidden_size, vocab_size))
        self.layers["b_head"] = np.empty((vocab_size))

        self.grad = {k: np.empty_like(v) for k, v in self.layers.items()}

        self.init_weights()

    @property
    def num_parameters(self):
        return sum(l.size for l in self.layers.values())



    def init_weights(self):
        for name, layer in self.layers.items():
            if layer.ndim == 1:
                self.layers[name] = np.zeros((layer.shape[0]))
            elif layer.ndim == 2:
                r, c = layer.shape
                d = np.sqrt(6.0 / (r + c))
                self.layers[name] = np.random.uniform(-d, d, (r, c))

    def init_state(self, batch_size):
        state = dict()

        state["h"] = np.zeros((self.n_cells, batch_size, self.hidden_size))
        state["c"] = np.zeros((self.n_cells, batch_size, self.hidden_size))
        return state

    def forward(
            self, inputs, album_id, state=None, teacher_forcing=True, generation_length=0
    ):
        batch_sz, seq_len = inputs.shape[:2]

        if teacher_forcing is True:
            assert generation_length == 0

        n_timestamps = seq_len + generation_length

        activations = defaultdict(lambda: defaultdict(list))
        activations["album_id"] = album_id

        outputs = np.zeros((batch_sz, n_timestamps, self.vocab_size))
        album_feats = self.layers["album_embedding"][album_id]

        if state is None:
            state = self.init_state(batch_sz)
        else:
            state = state.copy()  # make a shallow copy
        for k in ["h", "c"]:
            activations[k][-1] = state[k]

        for timestep in range(n_timestamps):

            if teacher_forcing is False and timestep >= 1:
                word_indices = np.argmax(outputs[:, timestep - 1], axis=1)
            else:
                word_indices = inputs[:, timestep]
            word_feats = self.layers["embedding"][word_indices]
            activations["input"][timestep] = word_indices

            features = np.concatenate((word_feats, album_feats), axis=-1)

            for cell_idx in range(self.n_cells):

                h_prev = state["h"][cell_idx]
                c_prev = state["c"][cell_idx]

                X = np.concatenate((features, h_prev), axis=-1)



                f = sigmoid(
                    X @ self.layers[f"Wf_{cell_idx}"]
                    + self.layers[f"bf_{cell_idx}"]
                )
                i = 1 - f
                o = sigmoid(
                    X @ self.layers[f"Wo_{cell_idx}"]
                    + self.layers[f"bo_{cell_idx}"]
                )
                c_bar = tanh(
                    X @ self.layers[f"Wc_{cell_idx}"]
                    + self.layers[f"bc_{cell_idx}"]
                )



                c = f * c_prev + i * c_bar
                h = o * tanh(c)

                if cell_idx == self.n_cells - 1:
                    logits = h @ self.layers["W_head"] + self.layers["b_head"]
                    probs = softmax(logits, axis=1)
                    outputs[:, timestep] = probs

                state["c"][cell_idx] = c
                state["h"][cell_idx] = h
                features = h

                for k, v in zip(
                        ["x", "f", "o", "c_bar", "c", "h"], [X, f, o, c_bar, c, h]
                ):
                    activations[k][timestep].append(v)
        return outputs, state, activations

    __call__ = forward

    def backward(self, grad, activations):
        batch_sz, seq_len = grad.shape[:2]

        grad_next = {
            k: np.zeros((self.n_cells, batch_sz, self.hidden_size))
            for k in ["h", "c"]
        }

        def _lin_grad(X, W, dY):
            return (dY @ W.T, X.T @ dY, dY)

        for timestep in reversed(range(seq_len)):

            dout_t = grad[:, timestep]
            h = activations["h"][timestep][-1]

            dh, dW_head, db_head = _lin_grad(
                X=h, W=self.layers["W_head"], dY=dout_t
            )
            self.grad[f"W_head"] += dW_head
            self.grad[f"b_head"] += np.sum(db_head, axis=0)

            for cell_idx in reversed(range(self.n_cells)):

                x, f, o, c_bar, c = (
                    activations[key][timestep][cell_idx]
                    for key in ["x", "f", "o", "c_bar", "c"]
                )

                c_prev = activations["c"][timestep - 1][cell_idx]

                dh += grad_next["h"][cell_idx]
                dc = grad_next["c"][cell_idx]

                do = dh * tanh(c)
                dc += dh * o * tanh(c, grad=True)

                df = dc * (c_prev - c_bar)
                dc_prev = dc * f
                dc_bar = dc * (1 - f)

                dc_bar *= tanh(c_bar, grad=True)
                do *= sigmoid(o, grad=True)
                df *= sigmoid(f, grad=True)

                dinp, dh_prev = 0, 0
                for gate, doutput in zip(["f", "o", "c"], [df, do, dc_bar]):
                    dX, dW, db = _lin_grad(
                        X=x, W=self.layers[f"W{gate}_{cell_idx}"], dY=doutput
                    )
                    self.grad[f"W{gate}_{cell_idx}"] += dW
                    self.grad[f"b{gate}_{cell_idx}"] += np.sum(db, axis=0)
                    dinp_gate, dh_prev_gate = (
                        dX[:, : -self.hidden_size],
                        dX[:, -self.hidden_size :],
                    )

                    dinp += dinp_gate
                    dh_prev += dh_prev_gate

                dh = dinp
                grad_next["c"][cell_idx] = dc_prev
                grad_next["h"][cell_idx] = dh_prev

            word_indices = activations["input"][timestep]
            d_word = dinp[:, :self.embed_size]
            d_album = dinp[:, self.embed_size:self.embed_size + self.album_embedding_size]
            self.grad["embedding"][word_indices] += d_word
            album_ids = activations["album_id"]
            self.grad["album_embedding"][album_ids] += d_album

    @property
    def state_dict(self):
        return dict(
            config=dict(
                embed_size=self.embed_size,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                n_cells=self.n_cells,
                album_count=self.album_count,
                album_embedding_size=self.album_embedding_size
            ),
            weights=deepcopy(self.layers),
            grad=deepcopy(self.grad),
        )
    @classmethod
    def from_state_dict(cls, state_dict):
        obj = cls(**state_dict["config"])
        for src, tgt in zip(
                [state_dict["weights"], state_dict["grad"]],
                [obj.layers, obj.grad],
        ):
            for k, v in src.items():
                tgt[k][:] = v
        return obj
