package regularizer_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/cronusio/gonn/pkg/regularizer"
)

type affineLayerMasker struct {
	forwardScale   float64
	forwardOffset  float64
	backwardScale  float64
	backwardOffset float64
}

func (m *affineLayerMasker) Penalty([]float64) float64                  { return 0 }
func (m *affineLayerMasker) WeightGrad(float64) float64                 { return 0 }
func (m *affineLayerMasker) ApplyMask(acts []float64, _ bool) []float64 { return acts }

func (m *affineLayerMasker) MaskForwardLayer(_ int, acts []float64) []float64 {
	for i := range acts {
		acts[i] = acts[i]*m.forwardScale + m.forwardOffset
	}
	return acts
}

func (m *affineLayerMasker) MaskBackwardLayer(_ int, upstream []float64) []float64 {
	for i := range upstream {
		upstream[i] = upstream[i]*m.backwardScale + m.backwardOffset
	}
	return upstream
}

func TestWeightGradientsAndIdentityMasks(t *testing.T) {
	l1 := regularizer.NewL1[float64](0.5)
	for _, tt := range []struct {
		weight float64
		want   float64
	}{{weight: 2, want: 0.5}, {weight: -2, want: -0.5}, {weight: 0, want: 0}} {
		if got := l1.WeightGrad(tt.weight); got != tt.want {
			t.Errorf("L1.WeightGrad(%v) = %v, want %v", tt.weight, got, tt.want)
		}
	}
	l2 := regularizer.NewL2[float64](0.25)
	if got := l2.WeightGrad(-2); got != -1 {
		t.Errorf("L2.WeightGrad(-2) = %v, want -1", got)
	}

	acts := []float64{1, -2, 3}
	for _, reg := range []regularizer.Regularizer[float64]{l1, l2} {
		got := reg.ApplyMask(acts, true)
		if !reflect.DeepEqual(got, acts) {
			t.Errorf("identity regularizer changed activations: got %v, want %v", got, acts)
		}
	}
}

func TestDropoutLayerMasks(t *testing.T) {
	const p = 0.5
	reg := regularizer.NewDropoutSeeded[float64](p, 42)
	acts := []float64{1, 1, 1, 1, 1, 1}
	forward := reg.MaskForwardLayer(2, acts)
	upstream := []float64{1, 1, 1, 1, 1, 1}
	backward := reg.MaskBackwardLayer(2, upstream)
	retained := 0
	for i := range forward {
		if forward[i] == 0 {
			if backward[i] != 0 {
				t.Errorf("dropped layer activation %d has gradient %v", i, backward[i])
			}
			continue
		}
		retained++
		if math.Abs(forward[i]-1/p) > 1e-12 {
			t.Errorf("retained layer activation %d = %v, want %v", i, forward[i], 1/p)
		}
		if math.Abs(backward[i]-1/p) > 1e-12 {
			t.Errorf("retained layer gradient %d = %v, want %v", i, backward[i], 1/p)
		}
	}
	if retained == 0 || retained == len(forward) {
		t.Fatalf("seeded mask retained %d of %d values; expected both retained and dropped values", retained, len(forward))
	}

	unmasked := []float64{2, 3}
	if got := reg.MaskBackwardLayer(9, unmasked); !reflect.DeepEqual(got, unmasked) {
		t.Errorf("backward pass without a sampled layer mask = %v, want %v", got, unmasked)
	}
}

func TestComposeLayerMaskOrderAndHelpers(t *testing.T) {
	first := &affineLayerMasker{forwardScale: 1, forwardOffset: 1, backwardScale: 1, backwardOffset: 1}
	second := &affineLayerMasker{forwardScale: 2, backwardScale: 2}
	composed := regularizer.Compose[float64](nil, first, regularizer.NewL2[float64](0.1), second)

	forward := composed.(regularizer.LayerMasker[float64]).MaskForwardLayer(0, []float64{1})
	if forward[0] != 4 {
		t.Errorf("composed forward masks = %v, want [4]", forward)
	}
	backward := composed.(regularizer.LayerMasker[float64]).MaskBackwardLayer(0, []float64{1})
	if backward[0] != 3 {
		t.Errorf("composed backward masks = %v, want [3]", backward)
	}

	empty := regularizer.Compose[float64]()
	input := []float64{1, 2}
	if got := empty.ApplyMask(input, true); !reflect.DeepEqual(got, input) {
		t.Errorf("empty composition changed activations: %v", got)
	}
	if got := empty.Penalty(input); got != 0 {
		t.Errorf("empty composition penalty = %v, want 0", got)
	}
	if got := empty.WeightGrad(3); got != 0 {
		t.Errorf("empty composition weight gradient = %v, want 0", got)
	}
	if got := empty.(regularizer.LayerMasker[float64]).MaskForwardLayer(0, input); !reflect.DeepEqual(got, input) {
		t.Errorf("empty composition forward mask = %v, want %v", got, input)
	}
	if got := empty.(regularizer.LayerMasker[float64]).MaskBackwardLayer(0, input); !reflect.DeepEqual(got, input) {
		t.Errorf("empty composition backward mask = %v, want %v", got, input)
	}
}

func TestApplyPenaltyAndAddWeightGrad(t *testing.T) {
	reg := regularizer.NewL2[float64](0.5)
	weights := []float64{1, -2, 3}
	grads := []float64{10, 20}
	regularizer.AddWeightGrad(reg, weights, grads)
	if !reflect.DeepEqual(grads, []float64{11, 18}) {
		t.Errorf("AddWeightGrad with shorter gradient slice = %v, want [11 18]", grads)
	}
	regularizer.AddWeightGrad[float64](nil, weights, grads)
	if !reflect.DeepEqual(grads, []float64{11, 18}) {
		t.Errorf("AddWeightGrad with nil regularizer changed gradients: %v", grads)
	}
	if got := regularizer.Penalty(reg, weights); got != 7 {
		t.Errorf("Penalty helper = %v, want 7", got)
	}
	acts := []float64{4, 5}
	if got := regularizer.Apply(reg, acts, true); !reflect.DeepEqual(got, acts) {
		t.Errorf("Apply helper changed identity-masked activations: %v", got)
	}
}
