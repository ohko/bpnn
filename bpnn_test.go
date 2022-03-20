package bpnn

import (
	"log"
	"testing"
)

func TestNewBPNN(t *testing.T) {
	log.SetFlags(log.Flags() | log.Lshortfile)

	nn, err := NewBPNN(2, 1, []int{2}, 0.6, 0.0001)
	if err != nil {
		t.Fatal(err)
	}

	nn.Hidden[0][0].Weight[0] = 0.1
	nn.Hidden[0][0].Weight[1] = -0.2
	nn.Hidden[0][1].Weight[0] = 0.4
	nn.Hidden[0][1].Weight[1] = 0.2
	nn.Output[0].Weight[0] = 0.2
	nn.Output[0].Weight[1] = -0.5

	result := nn.Train([]float64{0.4, -0.7}, []float64{0.1})
	log.Println(result)
}
