package ml

import (
	"fmt"
	"image"
	"log"

	onnx "github.com/owulveryck/onnx-go"
	"gorgonia.org/tensor"
)

type Model struct {
	onnxModel *onnx.Model
}

func NewModel(modelPath string) (*Model, error) {
	model := &onnx.Model{}
	file, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %w", err)
	}
	defer file.Close()

	if err := model.UnmarshalBinaryFrom(file); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX model: %w", err)
	}

	return &Model{onnxModel: model}, nil
}

func (m *Model) ExtractFeatures(img image.Image) ([]float32, error) {
	// Преобразование изображения в тензор
	t, err := imageToTensor(img)
	if err != nil {
		return nil, fmt.Errorf("failed to convert image to tensor: %w", err)
	}
	

	return []float32{}, nil
}

func imageToTensor(img image.Image) (tensor.Tensor, error) {
	// Преобразование изображения в тензор
	// Зависит от модели
	return nil, nil
}
