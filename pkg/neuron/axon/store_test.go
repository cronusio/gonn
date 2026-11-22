package axon

import "testing"

func TestStoreBindingPreservesWeights(t *testing.T) {
	var nilStore *Store[float64]
	if got := nilStore.Len(); got != 0 {
		t.Fatalf("nil Store.Len() = %d, want 0", got)
	}

	in := &stubNucleus[float64]{v: 2}
	out := &stubNeuron[float64]{}
	connection := NewWithWeight(0.25, in, out)

	store := &Store[float64]{W: make([]float64, 3)}
	if got := store.Len(); got != 3 {
		t.Fatalf("Store.Len() = %d, want 3", got)
	}
	connection.Bind(store, 1)
	if store.W[1] != 0.25 {
		t.Errorf("Bind copied weight %v, want 0.25", store.W[1])
	}

	store.W[1] = 0.5
	if got := connection.W(); got != 0.5 {
		t.Errorf("bound W() = %v after store update, want 0.5", got)
	}
	connection.SetW(0.75)
	if store.W[1] != 0.75 {
		t.Errorf("SetW wrote %v to store, want 0.75", store.W[1])
	}
	if got := connection.CalculateValue(); got != 1.5 {
		t.Errorf("bound CalculateValue() = %v, want 1.5", got)
	}

	rebuiltStore := &Store[float64]{W: make([]float64, 2)}
	connection.Bind(rebuiltStore, 0)
	if rebuiltStore.W[0] != 0.75 {
		t.Errorf("rebinding copied weight %v, want 0.75", rebuiltStore.W[0])
	}
	connection.Unbind()
	rebuiltStore.W[0] = 0.1
	if got := connection.W(); got != 0.75 {
		t.Errorf("Unbind did not preserve the live weight: got %v, want 0.75", got)
	}
	connection.SetW(1)
	if got := connection.W(); got != 1 {
		t.Errorf("unbound SetW() = %v, want 1", got)
	}

}
