package main

// How to use:
// pip install ultralytics
// go run create-standard-models.go

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/cyclopcam/cyclops/pkg/nn"

	"gopkg.in/yaml.v3"
)

type Size struct {
	Width  int
	Height int
}

type NCNNMetadata struct {
	Names map[int]string `yaml:"names"`
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func copyFile(dst, src string) error {
	os.MkdirAll(filepath.Dir(dst), 0755)
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	cerr := out.Close()
	if err != nil {
		return err
	}
	return cerr
}

func createNCNN() {
	versionVariants := []string{"v8", "11"}

	// 320x256 "m" feels like the sweet spot for CPU right now
	qualityVariants := []string{"n", "s", "m"}

	// CPU is just so slow at 640x480, that we leave this out for now.
	//sizeVariants := []Size{{320, 256}, {640, 480}}
	sizeVariants := []Size{{320, 256}}

	for _, v := range versionVariants {
		for _, q := range qualityVariants {
			for _, s := range sizeVariants {
				cmd := exec.Command("yolo", "export", "model=yolo"+v+q+".pt", "format=ncnn", "half=true", "imgsz="+fmt.Sprintf("%v,%v", s.Height, s.Width))
				//fmt.Printf("Exporting YOLO%v%v %v x %v to NCNN\n", v, q, s.Width, s.Height)
				fmt.Printf("%v\n", strings.Join(cmd.Args, " "))
				check(cmd.Run())
				outputDir := fmt.Sprintf("yolo%v%v_ncnn_model", v, q)
				metadataRaw, err := os.ReadFile(outputDir + "/metadata.yaml")
				check(err)
				metadata := NCNNMetadata{}
				check(yaml.Unmarshal(metadataRaw, &metadata))
				metaout := nn.ModelConfig{
					Architecture: "yolo" + v,
					Width:        s.Width,
					Height:       s.Height,
					Classes:      []string{},
				}
				for idx, class := range metadata.Names {
					for idx >= len(metaout.Classes) {
						metaout.Classes = append(metaout.Classes, "")
					}
					metaout.Classes[idx] = class
				}
				standardName := fmt.Sprintf("yolo%v%v_%v_%v", v, q, s.Width, s.Height)
				copyFile("coco/ncnn/"+standardName+".param", outputDir+"/model.ncnn.param")
				copyFile("coco/ncnn/"+standardName+".bin", outputDir+"/model.ncnn.bin")
				jm, err := json.MarshalIndent(&metaout, "", "\t")
				check(err)
				check(os.WriteFile("coco/ncnn/"+standardName+".json", jm, 0644))
				//os.Exit(0) // prototyping
			}
		}
	}
}

// Create ONNX models, which are the 1st step in creating hailo .hef models.
// The rest of the process runs inside the hailo container.
// I *tried* to get their various other mechanisms to work, but they all failed
// to install their Python packages.
func createONNX() {
	// I'm not seeing YOLO11 support yet from Hailo. Will wait for their official
	// support before even trying.
	versionVariants := []string{"v8"}
	qualityVariants := []string{"n", "s", "m", "l"}
	sizeVariants := []Size{{640, 640}}

	for _, v := range versionVariants {
		for _, q := range qualityVariants {
			for _, s := range sizeVariants {
				cmd := exec.Command("yolo", "export", "model=yolo"+v+q+".pt", "format=onnx", "imgsz="+fmt.Sprintf("%v,%v", s.Height, s.Width))
				fmt.Printf("%v\n", strings.Join(cmd.Args, " "))
				check(cmd.Run())
				//outputDir := fmt.Sprintf("yolo%v%v_ncnn_model", v, q)
				//metadataRaw, err := os.ReadFile(outputDir + "/metadata.yaml")
				//check(err)
			}
		}
	}
}

func main() {
	createNCNN()
	createONNX()
}
