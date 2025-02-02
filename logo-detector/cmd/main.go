package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/your-repo/logo-detector/internal/service"
	"github.com/your-repo/logo-detector/pkg/ml"
)

func main() {
	// Инициализация ML-модели
	modelPath := "models/logo_model.onnx"
	model, err := ml.NewModel(modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Создание сервиса
	logoService := service.NewLogoService(model)

	// Загрузка образцов логотипов
	sampleLogos := []string{"samples/logo1.png", "samples/logo2.png"}
	for _, sample := range sampleLogos {
		err := logoService.AddReferenceLogo(sample)
		if err != nil {
			log.Printf("Failed to add reference logo %s: %v", sample, err)
		}
	}

	// Тестирование на входном изображении
	inputImage := "samples/test_crop.png"
	isMatch, err := logoService.IsLogoMatch(context.Background(), inputImage)
	if err != nil {
		log.Fatalf("Error during logo matching: %v", err)
	}

	if isMatch {
		fmt.Println("The image contains the target logo.")
	} else {
		fmt.Println("The image does not contain the target logo.")
	}
}
