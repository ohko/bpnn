package bpnn

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
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
	for _, i := range hidden {
		tmp1 := make([]Neure, i)
		for k := range tmp1 {
			tmp1[k].Weight = make([]float64, len(o.Input))
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

// Train 训练
func (o *BPNN) Train(input, output []float64) bool {
	// 设置值
	for k, v := range input {
		o.Input[k].Value = v
	}
	for k, v := range output {
		o.Output[k].Value = v
	}

	logs := []string{}
	for nn := 0; nn < 1000; nn++ {
		// 计算隐藏层
		log.Println("计算隐藏层")
		for idx := range o.Hidden {
			for idx2 := range o.Hidden[idx] {
				sum := 0.0
				for idx3, in := range o.Input {
					sum += in.Value * o.Hidden[idx][idx2].Weight[idx3]
					log.Printf("%.2f += %.2f * %.2f\n", sum, in.Value, o.Hidden[idx][idx2].Weight[idx3])
				}
				o.Hidden[idx][idx2].Value = sigmoid(sum)
				log.Printf("sigmod(x)=%.2f", o.Hidden[idx][idx2].Value)
			}
		}

		// 计算输出层
		log.Println("计算输出层")
		last := len(o.Hidden) - 1
		for idx := range o.Output {
			sum := 0.0
			for idx2, hi := range o.Hidden[last] {
				sum += hi.Value * o.Output[idx].Weight[idx2]
				log.Printf("%.2f += %.2f * %.2f\n", sum, hi.Value, o.Output[idx].Weight[idx2])
			}
			o.Output[idx].NewValue = sigmoid(sum)
			log.Printf("sigmod(x)=%.2f", o.Output[idx].NewValue)
		}

		// 计算误差
		log.Println("计算误差")
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
		log.Printf("diff=%.2f", minDiff)

		// 阀值判断
		if minDiff < o.Diff {
			fmt.Println(strings.Join(logs, "\n"))
			return true
		}

		// 反向计算残差值
		log.Println("反向计算残差值")
		for idx := range o.Output {
			new, old := o.Output[idx].NewValue, o.Output[idx].Value
			o.Output[idx].ErrDiff = -(new - old) * new * (1 - new)
			log.Printf("errDiff=%.2f", o.Output[idx].ErrDiff)
		}

		// 计算输入层的误差
		// 输出 -> 最后一个隐藏层
		log.Println("输出 -> 最后一个隐藏层")
		for idx2 := range o.Hidden[last] {
			for idx3 := range o.Output {
				errDiff := o.Output[idx3].ErrDiff
				value := o.Hidden[last][idx2].Value
				weight := o.Output[idx3].Weight[idx2]
				o.Hidden[last][idx2].ErrDiff = (errDiff * weight) * value * (1 - value)
				o.Output[idx3].Weight[idx2] = weight - (value * -errDiff * o.Learn)
				log.Printf("误差:%.3f 修正：%.3f", o.Hidden[last][idx2].ErrDiff, o.Output[idx3].Weight[idx3])
			}
		}
		// 隐藏 -> 输入层
		for idx := range o.Hidden[0] {
			for idx2 := range o.Hidden[0][idx].Weight {
				errDiff := o.Hidden[0][idx].ErrDiff
				value := o.Input[idx2].Value
				weight := o.Hidden[0][idx].Weight[idx2]
				o.Hidden[0][idx].Weight[idx2] = weight - (value * -errDiff * o.Learn)
				log.Printf("误差:%.3f 修正：%.3f", errDiff, o.Hidden[0][idx2].Weight[idx])
			}
		}

		fmt.Printf("1->A:%.3f 2->A:%.3f\n1->B:%.3f 2->B:%.3f\nA->C:%.3f B->C:%.3f\n",
			o.Hidden[0][0].Weight[0],
			o.Hidden[0][0].Weight[1],
			o.Hidden[0][1].Weight[0],
			o.Hidden[0][1].Weight[1],
			o.Output[0].Weight[0],
			o.Output[0].Weight[1])
		logs = append(logs, fmt.Sprintf("%.8f", minDiff))
	}

	fmt.Println(strings.Join(logs, "\n"))
	return false
}

// 激活函数
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func randFloat64(min, max float64) float64 {
	if min == 0 && max == 0 {
		return 0
	}
	return rand.Float64()*(max-min) + min
}
