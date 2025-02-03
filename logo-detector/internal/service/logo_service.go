package service

import (
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/your-repo/logo-detector/pkg/ml"
)

type LogoService struct {
	model       *ml.Model
	referenceDB map[string][]float32 // Хранение 
}

func NewLogoService(model *ml.Model) *LogoService {
	return &LogoService{
		model:       model,
		referenceDB: make(map[string][]float32),
	}
}

// Добавление логотипа
func (s *LogoService) AddReferenceLogo(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return fmt.Errorf("failed to decode image: %w", err)
	}

	features, err := s.model.ExtractFeatures(img)
	if err != nil {
		return fmt.Errorf("failed to extract features: %w", err)
	}

	s.referenceDB[filePath] = features
	return nil
}

// Проверка соответствия входного изображения с логотипами
func (s *LogoService) IsLogoMatch(ctx context.Context, filePath string) (bool, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return false, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return false, fmt.Errorf("failed to decode image: %w", err)
	}

	inputFeatures, err := s.model.ExtractFeatures(img)
	if err != nil {
		return false, fmt.Errorf("failed to extract features: %w", err)
	}

	// Сравнение с samples 
	for _, refFeatures := range s.referenceDB {
		similarity := cosineSimilarity(inputFeatures, refFeatures)
		if similarity > 0.8 { // Пороговое значение
			return true, nil
		}
	}

	return false, nil
}

// Вычисление косинусного расстояния
func cosineSimilarity(vec1, vec2 []float32) float32 {
	var dotProduct, magnitude1, magnitude2 float32
	for i := range vec1 {
		dotProduct += vec1[i] * vec2[i]
		magnitude1 += vec1[i] * vec1[i]
		magnitude2 += vec2[i] * vec2[i]
	}
	return dotProduct / (sqrt(magnitude1) * sqrt(magnitude2))
}

func sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
