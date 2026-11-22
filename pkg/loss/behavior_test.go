package loss

import (
	"math"
	"testing"
)

func lossNear(got, want, tolerance float64) bool {
	return math.Abs(got-want) <= tolerance
}

func TestLossDispatch(t *testing.T) {
	const predicted, target = 0.8, 0.2
	difference := predicted - target
	logDifference := math.Log(predicted+1) - math.Log(target+1)
	tests := []struct {
		name string
		mode Type
		want float64
	}{
		{name: "mse", mode: MSE, want: 0.5 * difference * difference},
		{name: "mae", mode: MAE, want: difference},
		{name: "average", mode: AVG, want: difference},
		{name: "rmse scalar", mode: RMSE, want: difference},
		{name: "arctan", mode: ARCTAN, want: math.Atan(difference)},
		{name: "binary cross entropy", mode: BCE, want: -(target*math.Log(predicted) + (1-target)*math.Log(1-predicted))},
		{name: "mape", mode: MAPE, want: difference / target * 100},
		{name: "msle", mode: MSLE, want: logDifference * logDifference},
		{name: "kld", mode: KLD, want: target * math.Log(target/predicted)},
		{name: "categorical cross entropy", mode: CCE, want: -target * math.Log(predicted)},
		{name: "cross entropy alias", mode: CROSS_ENTROPY, want: -target * math.Log(predicted)},
		{name: "poisson", mode: POISSON, want: predicted - target*math.Log(predicted)},
		{name: "hinge", mode: HINGE, want: 1 - predicted*target},
		{name: "squared hinge", mode: SQ_HINGE, want: math.Pow(1-predicted*target, 2)},
		{name: "categorical hinge", mode: CAT_HINGE, want: 1 - predicted*target},
		{name: "log cosh", mode: LOG_COSH, want: math.Log(math.Cosh(difference))},
		{name: "huber", mode: HUBER, want: 0.5 * difference * difference},
		{name: "cosine scalar fallback", mode: COSINE, want: 0.5 * difference * difference},
		{name: "unknown fallback", mode: Type(99), want: 0.5 * difference * difference},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Loss(predicted, target, tt.mode); !lossNear(float64(got), tt.want, 1e-12) {
				t.Errorf("Loss(%v, %v, %v) = %.15g, want %.15g", predicted, target, tt.mode, got, tt.want)
			}
		})
	}
}

func TestLossClampsAndInactiveBranches(t *testing.T) {
	if got := Loss[float64](-1, 1, BCE); !lossNear(got, -math.Log(1e-7), 1e-12) {
		t.Errorf("BCE lower clamp = %v, want %v", got, -math.Log(1e-7))
	}
	if got := Loss[float64](2, 0, BCE); !lossNear(got, -math.Log(1e-7), 1e-8) {
		t.Errorf("BCE upper clamp = %v, want %v", got, -math.Log(1e-7))
	}
	if got := Loss[float64](0.5, 0, KLD); got != 0 {
		t.Errorf("KLD with a zero target = %v, want 0", got)
	}
	if got := Loss[float64](-2, 1, KLD); !lossNear(got, -math.Log(1e-7), 1e-12) {
		t.Errorf("KLD lower clamp = %v, want %v", got, -math.Log(1e-7))
	}
	if got := Loss[float64](-2, 1, POISSON); !lossNear(got, 1e-7-math.Log(1e-7), 1e-12) {
		t.Errorf("Poisson lower clamp = %v", got)
	}
	if got := Loss[float64](0.5, 0, MAPE); !lossNear(got, 5e8, 1e-6) {
		t.Errorf("MAPE zero-target clamp = %v, want 500000000", got)
	}
	if got := Loss[float64](2, 1, HINGE); got != 0 {
		t.Errorf("inactive hinge loss = %v, want 0", got)
	}
	if got := Loss[float64](2, 1, SQ_HINGE); got != 0 {
		t.Errorf("inactive squared hinge loss = %v, want 0", got)
	}
	if got := Loss[float64](1, 0, CCE); got != 0 {
		t.Errorf("CCE with zero target = %v, want 0", got)
	}
	if got := Loss[float64](0, 1, CCE); !lossNear(got, -math.Log(1e-7), 1e-12) {
		t.Errorf("CCE lower clamp = %v", got)
	}
	if got := huberLossWithDelta(3.0, 0.0, 2.0); got != 4 {
		t.Errorf("Huber custom delta = %v, want 4", got)
	}
}

func TestLossDerivatives(t *testing.T) {
	const predicted, target = 0.8, 0.2
	difference := predicted - target
	tests := []struct {
		name string
		mode Type
		want float64
	}{
		{name: "mse", mode: MSE, want: difference},
		{name: "mae", mode: MAE, want: 1},
		{name: "average", mode: AVG, want: 1},
		{name: "rmse", mode: RMSE, want: difference},
		{name: "bce", mode: BCE, want: difference / (predicted * (1 - predicted))},
		{name: "msle", mode: MSLE, want: 2 * (math.Log(predicted+1) - math.Log(target+1)) / (predicted + 1)},
		{name: "kld", mode: KLD, want: -target / predicted},
		{name: "poisson", mode: POISSON, want: 1 - target/predicted},
		{name: "mape", mode: MAPE, want: 100 / target},
		{name: "hinge", mode: HINGE, want: -target},
		{name: "squared hinge", mode: SQ_HINGE, want: -2 * target * (1 - predicted*target)},
		{name: "categorical hinge", mode: CAT_HINGE, want: -target},
		{name: "log cosh", mode: LOG_COSH, want: math.Tanh(difference)},
		{name: "huber quadratic", mode: HUBER, want: difference},
		{name: "arctan", mode: ARCTAN, want: 1 / (1 + difference*difference)},
		{name: "categorical cross entropy", mode: CCE, want: -target / predicted},
		{name: "cross entropy alias", mode: CROSS_ENTROPY, want: -target / predicted},
		{name: "cosine fallback", mode: COSINE, want: difference},
		{name: "unknown fallback", mode: Type(99), want: 2 * difference},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Derivative(predicted, target, tt.mode); !lossNear(float64(got), tt.want, 1e-12) {
				t.Errorf("Derivative(%v, %v, %v) = %.15g, want %.15g", predicted, target, tt.mode, got, tt.want)
			}
		})
	}

	if got := Derivative[float64](0.5, 0, KLD); got != 0 {
		t.Errorf("KLD derivative with zero target = %v, want 0", got)
	}
	if got := Derivative[float64](0, 0, MAPE); got != 0 {
		t.Errorf("MAPE derivative at zero error = %v, want 0", got)
	}
	if got := Derivative[float64](0.5, 0, MAPE); !lossNear(got, 1e9, 1e-6) {
		t.Errorf("MAPE derivative zero-target clamp = %v", got)
	}
	if got := Derivative[float64](0, 1, BCE); !lossNear(got, (1e-7-1)/(1e-7*(1-1e-7)), 1e-3) {
		t.Errorf("BCE derivative lower clamp = %v", got)
	}
	if got := Derivative[float64](1, 0, BCE); !lossNear(got, (1-1e-7)/((1-1e-7)*1e-7), 1e-2) {
		t.Errorf("BCE derivative upper clamp = %v", got)
	}
	if got := Derivative[float64](2, 1, HINGE); got != 0 {
		t.Errorf("inactive hinge derivative = %v, want 0", got)
	}
	if got := Derivative[float64](2, 1, SQ_HINGE); got != 0 {
		t.Errorf("inactive squared hinge derivative = %v, want 0", got)
	}
	if got := Derivative[float64](3, 0, HUBER); got != 1 {
		t.Errorf("Huber linear derivative = %v, want 1", got)
	}
	clampedMSLEInput := -1.0 + 1e-7
	if got := Derivative[float64](-2, 0, MSLE); !lossNear(got, 2*math.Log(clampedMSLEInput+1)/(clampedMSLEInput+1), 1e-5) {
		t.Errorf("MSLE derivative at lower clamp = %v", got)
	}
}

func TestVectorLossAndDerivative(t *testing.T) {
	predicted := []float64{0.2, 0.3, 0.5}
	target := []float64{0, 0, 1}
	wantCCE := -math.Log(0.5)
	if got := VectorLoss(predicted, target, CCE); !lossNear(got, wantCCE, 1e-12) {
		t.Errorf("VectorLoss CCE = %v, want %v", got, wantCCE)
	}
	if got := VectorLoss(predicted, target, CROSS_ENTROPY); !lossNear(got, wantCCE, 1e-12) {
		t.Errorf("VectorLoss cross entropy alias = %v, want %v", got, wantCCE)
	}
	if got := VectorLoss([]float64{1, 0}, []float64{1, 0}, COSINE); !lossNear(got, 0, 1e-12) {
		t.Errorf("VectorLoss cosine identical vectors = %v, want 0", got)
	}
	if got := VectorLoss([]float64{0, 0}, []float64{0, 0}, COSINE); got != 1 {
		t.Errorf("VectorLoss cosine zero vectors = %v, want 1", got)
	}
	if got := VectorLoss([]float64{0.1, 0.9}, []float64{1, 0}, CAT_HINGE); !lossNear(got, 1.8, 1e-12) {
		t.Errorf("VectorLoss categorical hinge = %v, want 1.8", got)
	}
	if got := VectorLoss([]float64{2, 0}, []float64{1, 0}, CAT_HINGE); got != 0 {
		t.Errorf("VectorLoss inactive categorical hinge = %v, want 0", got)
	}
	if got := VectorLoss([]float64{0.5, 0.5}, []float64{1, 0}, MSE); !lossNear(got, 0.25, 1e-12) {
		t.Errorf("VectorLoss element-wise fallback = %v, want 0.25", got)
	}

	gotCCE := VectorDerivative(predicted, target, CCE)
	wantCCEGradient := []float64{0, 0, -2}
	for i := range gotCCE {
		if !lossNear(gotCCE[i], wantCCEGradient[i], 1e-12) {
			t.Errorf("VectorDerivative CCE[%d] = %v, want %v", i, gotCCE[i], wantCCEGradient[i])
		}
	}
	gotAlias := VectorDerivative(predicted, target, CROSS_ENTROPY)
	if gotAlias[2] != -2 {
		t.Errorf("VectorDerivative cross entropy alias = %v, want final gradient -2", gotAlias)
	}
	gotHinge := VectorDerivative([]float64{0.1, 0.9}, []float64{1, 0}, CAT_HINGE)
	if gotHinge[0] != -1 || gotHinge[1] != 1 {
		t.Errorf("VectorDerivative categorical hinge = %v, want [-1 1]", gotHinge)
	}
	gotNoClass := VectorDerivative([]float64{0.1, 0.9}, []float64{0, 0}, CAT_HINGE)
	if gotNoClass[0] != 1 || gotNoClass[1] != 1 {
		t.Errorf("VectorDerivative categorical hinge without a positive target = %v, want [1 1]", gotNoClass)
	}
	gotElementwise := VectorDerivative([]float64{0.5, 1}, []float64{0, 0.25}, MSE)
	if gotElementwise[0] != 0.5 || gotElementwise[1] != 0.75 {
		t.Errorf("VectorDerivative element-wise fallback = %v, want [0.5 0.75]", gotElementwise)
	}
	if gotEmpty := VectorDerivative[float64](nil, nil, CCE); len(gotEmpty) != 0 {
		t.Errorf("VectorDerivative of empty vectors = %v, want empty", gotEmpty)
	}
}

func TestAggregateAndCalculateTotalLoss(t *testing.T) {
	if got := Aggregate([]float64{1, 3}, []float64{0, 1}, MSE); got != 2.5 {
		t.Errorf("Aggregate MSE = %v, want 2.5", got)
	}
	if got := Aggregate([]float64{1, 3}, []float64{0, 1}, RMSE); !lossNear(got, math.Sqrt(2.5), 1e-12) {
		t.Errorf("Aggregate RMSE = %v, want %v", got, math.Sqrt(2.5))
	}
	if got := Aggregate([]float64{0.2, 0.8}, []float64{1, 0}, CCE); !lossNear(got, -math.Log(0.2), 1e-12) {
		t.Errorf("Aggregate CCE = %v", got)
	}
	if got := Aggregate([]float64{1, 0}, []float64{1, 0}, COSINE); got != 0 {
		t.Errorf("Aggregate cosine = %v, want 0", got)
	}
	if got := Aggregate([]float64{0.1, 0.9}, []float64{1, 0}, CAT_HINGE); !lossNear(got, 1.8, 1e-12) {
		t.Errorf("Aggregate categorical hinge = %v, want 1.8", got)
	}
	if got := Aggregate[float64](nil, nil, RMSE); got != 0 {
		t.Errorf("Aggregate of empty vectors = %v, want 0", got)
	}

	first, second := 2.0, 4.0
	misses := []*float64{&first, &second}
	if got := CalculateTotalLoss(&misses, MSE); got != 5 {
		t.Errorf("CalculateTotalLoss mean = %v, want 5", got)
	}
	single := []*float64{&first}
	if got := CalculateTotalLoss(&single, MSE); got != 2 {
		t.Errorf("CalculateTotalLoss single sample = %v, want 2", got)
	}
	empty := []*float64{}
	if got := CalculateTotalLoss(&empty, MSE); got != 0 {
		t.Errorf("CalculateTotalLoss empty batch = %v, want 0", got)
	}
	for _, mode := range []Type{RMSE, MSLE, LOG_COSH, HUBER} {
		if got := CalculateTotalLoss(&misses, mode); math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
			t.Errorf("CalculateTotalLoss(%v) is not finite: %v", mode, got)
		}
	}
}
