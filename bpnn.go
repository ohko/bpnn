package bpnn

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
)

type BPNN struct {
	Input  []Neure   // 输入层
	Hidden [][]Neure // 隐藏层
	Output []Neure   // 输出层
	Learn  float64   // 学习速率
	Diff   float64   // 最小误差阀值
}

// Neure 神经元
type Neure struct {
	Value    float64   // 原值
	NewValue float64   // 新值
	Weight   []float64 // 权重
	Diff     float64   // 误差
	ErrDiff  float64   // 残差
}

// NewBPNN 创建BPNN
// hidden: [2,3]
func NewBPNN(inputCount, outputCount int, hidden []int, learn, diff float64) (*BPNN, error) {
	o := &BPNN{
		Input:  make([]Neure, inputCount),
		Output: make([]Neure, outputCount),
		Learn:  learn,
		Diff:   diff,
	}

	// 初始化输入层权重
	for i := range o.Input {
		o.Input[i].Weight = []float64{randFloat64(0, 1)}
	}

	// 初始化隐藏层
	tmp := [][]Neure{}
	for m, i := range hidden {
		tmp1 := make([]Neure, i)
		for k := range tmp1 {
			if m == 0 {
				tmp1[k].Weight = make([]float64, len(o.Input))
			} else {
				tmp1[k].Weight = make([]float64, hidden[m-1])
			}
			for j := 0; j < len(o.Input); j++ {
				tmp1[k].Weight[j] = randFloat64(0, 1)
			}
		}
		tmp = append(tmp, tmp1)
	}
	o.Hidden = tmp

	// 初始化输出层权重
	last := len(o.Hidden) - 1
	for i := range o.Output {
		o.Output[i].Weight = make([]float64, len(o.Hidden[last]))
		for j := 0; j < len(o.Hidden[last]); j++ {
			o.Output[i].Weight[j] = randFloat64(0, 1)
		}
	}

	return o, nil
}

// 计算模块
func calcForward(input, output []Neure, last bool) {
	for idxInput := range output {
		sum := 0.0
		for idxOutput := range input {
			sum += input[idxOutput].Value * output[idxInput].Weight[idxOutput]
			// log.Printf("%.3f += %.3f * %.3f\n", sum, input[idxOutput].Value, output[idxInput].Weight[idxOutput])
		}
		if last {
			output[idxInput].NewValue = sigmoid(sum)
			// log.Printf("sigmod(x)=%.3f", output[idxInput].NewValue)
		} else {
			output[idxInput].Value = sigmoid(sum)
			// log.Printf("sigmod(x)=%.3f", output[idxInput].Value)
		}
	}
}
func calcBackward(learn float64, input, output []Neure) {
	for idxInput := range input {
		for idxOutput := range input[idxInput].Weight {
			output[idxOutput].ErrDiff = 0
		}
	}
	for idxInput := range input {
		for idxOutput := range input[idxInput].Weight {
			errDiff := input[idxInput].ErrDiff
			value := output[idxOutput].Value
			weight := input[idxInput].Weight[idxOutput]
			input[idxInput].Weight[idxOutput] = weight - (value * -errDiff * learn)
			output[idxOutput].ErrDiff += (errDiff * weight) * value * (1 - value)
			// log.Printf("weight: %.3f - (%.3f * -%.3f * %.3f)\n", weight, value, errDiff, learn)
			// log.Printf("(%.3f * %.3f) * %.3f * (1 - %.3f)\n", errDiff, weight, value, value)
			// log.Printf("误差:%.3f 残差和：%.3f 修正：%.3f\n", errDiff, output[idxOutput].ErrDiff, input[idxInput].Weight[idxOutput])
		}
	}
}

// Train 训练
func (o *BPNN) Train(input, output [][]float64, trainCount int) (float64, int) {
	min := 0.0
	for mm := 0; mm < trainCount; mm++ {
		min = 0.0
		for i := range input {
			result := o.train(input[i], output[i])
			if min == 0 {
				min = result
			}
			if result < min {
				min = result
			}
		}
		if min < 0.0001 {
			return min, mm
		}
		log.Println(min)
	}
	return min, trainCount
}

// Check 检测
func (o *BPNN) Check(input []float64) []float64 {
	// 设置值
	for k, v := range input {
		o.Input[k].Value = v
	}

	// for nn := 0; nn < 1000; nn++ {
	// 计算输入层->隐藏层
	// log.Println("计算输入层->隐藏层")
	calcForward(o.Input, o.Hidden[0], false)

	// 计算隐藏层->隐藏层
	// log.Println("计算隐藏层->隐藏层")
	last := len(o.Hidden) - 1
	if last >= 1 {
		for idx := range o.Hidden[0:last] {
			calcForward(o.Hidden[idx], o.Hidden[idx+1], false)
		}
	}

	// 计算隐藏层->输出层
	// log.Println("计算隐藏层->输出层")
	calcForward(o.Hidden[last], o.Output, true)

	output := []float64{}
	for i := range o.Output {
		output = append(output, o.Output[i].NewValue)
	}

	return output
}

func (o *BPNN) Export() string {
	bs, _ := json.Marshal(o)
	fmt.Println(string(bs))
	return string(bs)
}
func (o *BPNN) Import(data string) error {
	return json.Unmarshal([]byte(data), o)
}
func NewFromJSON(data string) (*BPNN, error) {
	var o BPNN
	return &o, json.Unmarshal([]byte(data), &o)
}

func (o *BPNN) train(input, output []float64) float64 {
	// 设置值
	for k, v := range input {
		o.Input[k].Value = v
	}
	for k, v := range output {
		o.Output[k].Value = v
	}

	// for nn := 0; nn < 1000; nn++ {
	// 计算输入层->隐藏层
	// log.Println("计算输入层->隐藏层")
	calcForward(o.Input, o.Hidden[0], false)

	// 计算隐藏层->隐藏层
	// log.Println("计算隐藏层->隐藏层")
	last := len(o.Hidden) - 1
	if last >= 1 {
		for idx := range o.Hidden[0:last] {
			calcForward(o.Hidden[idx], o.Hidden[idx+1], false)
		}
	}

	// 计算隐藏层->输出层
	// log.Println("计算隐藏层->输出层")
	calcForward(o.Hidden[last], o.Output, true)

	// 计算误差
	// log.Println("计算误差")
	minDiff := 0.0
	for idx := range o.Output {
		o.Output[idx].Diff = math.Pow(o.Output[idx].NewValue-o.Output[idx].Value, 2)
		if minDiff == 0.0 {
			minDiff = o.Output[idx].Diff
		}
		if o.Output[idx].Diff < minDiff {
			minDiff = o.Output[idx].Diff
		}
	}
	// log.Printf("diff=%.3f", minDiff)

	// 阀值判断
	if minDiff < o.Diff {
		return minDiff
	}

	// 反向计算残差值
	// log.Println("反向计算残差值")
	for idx := range o.Output {
		new, old := o.Output[idx].NewValue, o.Output[idx].Value
		o.Output[idx].ErrDiff = -(new - old) * new * (1 - new)
		// log.Printf("errDiff=%.3f", o.Output[idx].ErrDiff)
	}

	// 计算输入层的误差
	// 输出 -> 最后一个隐藏层
	// log.Println("输出 -> 最后一个隐藏层")
	calcBackward(o.Learn, o.Output, o.Hidden[last])

	// 隐藏 -> 隐藏层
	// log.Println("隐藏 -> 隐藏层")
	if last >= 1 {
		for idx := last; idx > 0; idx-- {
			calcBackward(o.Learn, o.Hidden[idx], o.Hidden[idx-1])
		}
	}

	// 隐藏 -> 输入层
	// log.Println("隐藏 -> 输入层")
	calcBackward(o.Learn, o.Hidden[0], o.Input)
	// }

	return minDiff
}

// 激活函数
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func randFloat64(min, max float64) float64 {
	if min == 0 && max == 0 {
		return 0
	}
	// rand.Seed(time.Now().UnixNano())
	return rand.Float64()*(max-min) + min
}
