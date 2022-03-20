package bpnn

import (
	"log"
	"testing"
)

// go test -timeout 15s -run ^TestNewBPNN$ bpnn -v -count=1
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

	result, _ := nn.Train([][]float64{{0.4, -0.7}}, [][]float64{{0.1}}, 1000)
	log.Println(result)
}

// go test -timeout 15s -run ^TestCheck$ bpnn -v -count=1
func TestCheck(t *testing.T) {
	log.SetFlags(log.Flags() | log.Lshortfile)

	nn, err := NewBPNN(2, 2, []int{3, 3}, 0.6, 0.0001)
	if err != nil {
		t.Fatal(err)
	}

	nn.Hidden[0][0].Weight[0] = 0.2
	nn.Hidden[0][0].Weight[1] = 0.5
	nn.Hidden[0][1].Weight[0] = 0.3
	nn.Hidden[0][1].Weight[1] = 0.6
	nn.Hidden[0][2].Weight[0] = 0.4
	nn.Hidden[0][2].Weight[1] = 0.7
	nn.Hidden[1][0].Weight[0] = 0.1
	nn.Hidden[1][0].Weight[1] = 0.4
	nn.Hidden[1][0].Weight[2] = 0.7
	nn.Hidden[1][1].Weight[0] = 0.2
	nn.Hidden[1][1].Weight[1] = 0.5
	nn.Hidden[1][1].Weight[2] = 0.8
	nn.Hidden[1][2].Weight[0] = 0.3
	nn.Hidden[1][2].Weight[1] = 0.6
	nn.Hidden[1][2].Weight[2] = 0.9
	nn.Output[0].Weight[0] = 0.2
	nn.Output[0].Weight[1] = 0.6
	nn.Output[0].Weight[2] = 0.3
	nn.Output[1].Weight[0] = 0.4
	nn.Output[1].Weight[1] = 0.8
	nn.Output[1].Weight[2] = 0.5

	result, _ := nn.Train([][]float64{{0.1, 0.2}}, [][]float64{{0.3, 0.4}}, 1000)
	log.Println(result)
}

// go test -timeout 15s -run ^Test_1$ bpnn -v -count=1
func Test_1(t *testing.T) {
	log.SetFlags(log.Flags() | log.Lshortfile)

	nn, err := NewBPNN(2, 1, []int{3, 3}, 0.6, 0.0001)
	if err != nil {
		t.Fatal(err)
	}

	inputData := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	outputData := [][]float64{
		{1},
		{0},
		{0},
		{1},
	}

	min, count := nn.Train(inputData, outputData, 100000)
	log.Printf("min:%.8f count:%d\n", min, count)

	jsonData := nn.Export()
	nn.Import(jsonData)
	log.Println(nn.Check([]float64{0, 0}))
	log.Println(nn.Check([]float64{0, 1}))
	log.Println(nn.Check([]float64{1, 0}))
	log.Println(nn.Check([]float64{1, 1}))

	newNN, err := NewFromJSON(jsonData)
	if err != nil {
		t.Fatal(err)
	}
	log.Println(newNN.Check([]float64{0, 0}))
	log.Println(newNN.Check([]float64{0, 1}))
	log.Println(newNN.Check([]float64{1, 0}))
	log.Println(newNN.Check([]float64{1, 1}))
}
