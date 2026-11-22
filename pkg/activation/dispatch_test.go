package activation

import (
	"math"
	"testing"
)

func activationNear(got, want, tolerance float64) bool {
	return math.Abs(got-want) <= tolerance
}

func TestActivationAndDerivativeDispatch(t *testing.T) {
	tests := []struct {
		name       string
		mode       Type
		value      float64
		params     []float64
		activation float64
		derivative float64
	}{
		{name: "elish-positive", mode: ELISH, value: 1, activation: 1 / (1 + math.Exp(-1)), derivative: 1/(1+math.Exp(-1)) + math.Exp(-1)/math.Pow(1+math.Exp(-1), 2)},
		{name: "elish-negative", mode: ELISH, value: -1, activation: (math.Exp(-1) - 1) / (1 + math.E), derivative: math.Exp(-1)/(1+math.E) + (math.Exp(-1)-1)*math.E/math.Pow(1+math.E, 2)},
		{name: "elu-negative", mode: ELU, value: -1, params: []float64{2}, activation: 2 * (math.Exp(-1) - 1), derivative: 2 * math.Exp(-1)},
		{name: "elu-positive", mode: ELU, value: 2, activation: 2, derivative: 1},
		{name: "linear-custom", mode: Linear, value: 2, params: []float64{3, -1}, activation: 5, derivative: 3},
		{name: "leaky-relu-negative", mode: LeakyReLU, value: -2, params: []float64{0.2}, activation: -0.4, derivative: 0.2},
		{name: "relu-negative", mode: ReLU, value: -2, activation: 0, derivative: 0},
		{name: "relu-positive", mode: ReLU, value: 2, activation: 2, derivative: 1},
		{name: "selu-negative", mode: SELU, value: -1, params: []float64{2, 1.5}, activation: 3 * (math.Exp(-1) - 1), derivative: 3 * math.Exp(-1)},
		{name: "selu-positive", mode: SELU, value: 2, params: []float64{2, 1.5}, activation: 4, derivative: 2},
		{name: "sigmoid-custom", mode: SIGMOID, value: 0.5, params: []float64{2}, activation: 1 / (1 + math.Exp(-1)), derivative: 2 * (1 / (1 + math.Exp(-1))) * (1 - 1/(1+math.Exp(-1)))},
		{name: "softmax-scalar", mode: SOFTMAX, value: -0.5, activation: 1 / (1 + math.Exp(0.5)), derivative: (1 / (1 + math.Exp(0.5))) * (1 - 1/(1+math.Exp(0.5)))},
		{name: "swish", mode: SWISH, value: 1, params: []float64{1}, activation: 1 / (1 + math.Exp(-1)), derivative: 1/(1+math.Exp(-1)) + (1/(1+math.Exp(-1)))*(1-1/(1+math.Exp(-1)))},
		{name: "tanh", mode: TanH, value: -1, activation: math.Tanh(-1), derivative: 1 - math.Pow(math.Tanh(-1), 2)},
		{name: "unknown", mode: Type(99), value: 2.5, activation: 2.5, derivative: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Activation(tt.value, tt.mode, tt.params...); !activationNear(float64(got), tt.activation, 1e-12) {
				t.Errorf("Activation(%v, %v) = %v, want %v", tt.value, tt.mode, got, tt.activation)
			}
			if got := Derivative(tt.value, tt.mode, tt.params...); !activationNear(float64(got), tt.derivative, 1e-12) {
				t.Errorf("Derivative(%v, %v) = %v, want %v", tt.value, tt.mode, got, tt.derivative)
			}
		})
	}
}

func TestActivationDefaultsAndStatefulSigmoid(t *testing.T) {
	if got := Activation[float64](2, Linear); got != 2 {
		t.Errorf("Activation with default linear parameters = %v, want 2", got)
	}
	if got := Derivative[float64](2, Linear); got != 1 {
		t.Errorf("Derivative with default linear parameters = %v, want 1", got)
	}
	if got := Activation[float64](-2, LeakyReLU); got != -0.02 {
		t.Errorf("Activation with default leak = %v, want -0.02", got)
	}
	if got := Derivative[float64](-2, LeakyReLU); got != 0.01 {
		t.Errorf("Derivative with default leak = %v, want 0.01", got)
	}
	if got := Activation[float64](-1, ELU); !activationNear(got, math.Exp(-1)-1, 1e-12) {
		t.Errorf("Activation with default ELU alpha = %v", got)
	}
	if got := Activation[float64](1, SELU); !activationNear(got, 1.0507009873554805, 1e-12) {
		t.Errorf("Activation with default SELU parameters = %v", got)
	}
	if got := Derivative[float64](1, SELU); !activationNear(got, 1.0507009873554805, 1e-12) {
		t.Errorf("Derivative with default SELU parameters = %v", got)
	}
	if got := Activation[float64](1, SWISH); !activationNear(got, 1/(1+math.Exp(-1)), 1e-12) {
		t.Errorf("Activation with default swish beta = %v", got)
	}
	if got := Derivative[float64](1, SWISH); !activationNear(got, 1/(1+math.Exp(-1))+(1/(1+math.Exp(-1)))*(1-1/(1+math.Exp(-1))), 1e-12) {
		t.Errorf("Derivative with default swish beta = %v", got)
	}
	if got := Activation[float64](1, SIGMOID); !activationNear(got, 1/(1+math.Exp(-1)), 1e-12) {
		t.Errorf("Activation with default sigmoid slope = %v", got)
	}
	if got := Derivative[float64](1, SIGMOID); !activationNear(got, (1/(1+math.Exp(-1)))*(1-1/(1+math.Exp(-1))), 1e-12) {
		t.Errorf("Derivative with default sigmoid slope = %v", got)
	}

	activation := NewSigmoid[float64](2)
	value := 0.5
	activation.Activation(&value)
	want := 1 / (1 + math.Exp(-1))
	if !activationNear(value, want, 1e-12) {
		t.Fatalf("Sigmoid.Activation() = %v, want %v", value, want)
	}
	activation.Derivative(&value)
	if !activationNear(value, 2*want*(1-want), 1e-12) {
		t.Errorf("Sigmoid.Derivative() = %v, want %v", value, 2*want*(1-want))
	}
	if got := SigmoidActivation(0.5, 2.0); !activationNear(got, want, 1e-12) {
		t.Errorf("SigmoidActivation() = %v, want %v", got, want)
	}
	if got := SigmoidDerivative(0.5, 2.0); !activationNear(got, 2*want*(1-want), 1e-12) {
		t.Errorf("SigmoidDerivative() = %v, want %v", got, 2*want*(1-want))
	}
}

func TestSoftmaxInto(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
		want  []float64
	}{
		{name: "probability distribution", input: []float64{1, 2, 3}, want: []float64{0.09003057317038046, 0.24472847105479764, 0.6652409557748218}},
		{name: "large logits", input: []float64{1000, 1001}, want: []float64{0.2689414213699951, 0.7310585786300049}},
		{name: "single element", input: []float64{42}, want: []float64{1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := make([]float64, len(tt.input))
			SoftmaxInto(got, tt.input)
			var sum float64
			for i := range got {
				if !activationNear(got[i], tt.want[i], 1e-12) {
					t.Errorf("SoftmaxInto()[%d] = %v, want %v", i, got[i], tt.want[i])
				}
				sum += got[i]
			}
			if !activationNear(sum, 1, 1e-12) {
				t.Errorf("softmax probabilities sum to %v, want 1", sum)
			}
		})
	}

	aliased := []float64{1, 2, 3}
	SoftmaxInto(aliased, aliased)
	if !activationNear(aliased[0], 0.09003057317038046, 1e-12) || !activationNear(aliased[2], 0.6652409557748218, 1e-12) {
		t.Errorf("SoftmaxInto did not support aliased input and output: %v", aliased)
	}
	SoftmaxInto[float64](nil, nil)
}
