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

type variant struct {
	engine  string
	version string
	quality string
	size    Size
}

func createVariants(engine string, versions []string, qualities []string, sizes []Size) []variant {
	variants := []variant{}
	for _, v := range versions {
		for _, q := range qualities {
			for _, s := range sizes {
				variants = append(variants, variant{engine, v, q, s})
			}
		}
	}
	return variants
}

func createNCNN() {
	versionVariants := []string{"v8", "11"}

	// 320x256 "m" feels like the sweet spot for CPU right now
	qualityVariants := []string{"n", "s", "m"}

	// CPU is just so slow at 640x480, that we leave this out for now.
	sizeVariants := []Size{{320, 256}}

	variants := createVariants("ncnn", versionVariants, qualityVariants, sizeVariants)

	// We also want a 640x480 NCNN yolov8m, because this allows us to compare it to hailo 8L models.
	// If the NCNN 640x480 yolov8m model is more accurate than the hailo 8L model, then we can use
	// this as additional verification before firing off an alarm. The Hailo models are 640x640,
	// but I think 640x480 is fine for apples-to-apples, and 640x640 is just a waste of precious
	// CPU inference cycles.
	variants = append(variants, variant{"ncnn", "v8", "m", Size{640, 480}})
	variants = append(variants, variant{"ncnn", "v8", "l", Size{640, 480}})

	for _, v := range variants {
		cmd := exec.Command("yolo", "export", "model=yolo"+v.version+v.quality+".pt", "format=ncnn", "half=true", "imgsz="+fmt.Sprintf("%v,%v", v.size.Height, v.size.Width))
		//fmt.Printf("Exporting YOLO%v%v %v x %v to NCNN\n", v, q, s.Width, s.Height)
		fmt.Printf("%v\n", strings.Join(cmd.Args, " "))
		check(cmd.Run())
		outputDir := fmt.Sprintf("yolo%v%v_ncnn_model", v.version, v.quality)
		metadataRaw, err := os.ReadFile(outputDir + "/metadata.yaml")
		check(err)
		metadata := NCNNMetadata{}
		check(yaml.Unmarshal(metadataRaw, &metadata))
		metaout := nn.ModelConfig{
			Architecture: "yolo" + v.version,
			Width:        v.size.Width,
			Height:       v.size.Height,
			Classes:      []string{},
		}
		for idx, class := range metadata.Names {
			for idx >= len(metaout.Classes) {
				metaout.Classes = append(metaout.Classes, "")
			}
			metaout.Classes[idx] = class
		}
		standardName := fmt.Sprintf("yolo%v%v_%v_%v", v.version, v.quality, v.size.Width, v.size.Height)
		copyFile("coco/ncnn/"+standardName+".param", outputDir+"/model.ncnn.param")
		copyFile("coco/ncnn/"+standardName+".bin", outputDir+"/model.ncnn.bin")
		jm, err := json.MarshalIndent(&metaout, "", "\t")
		check(err)
		check(os.WriteFile("coco/ncnn/"+standardName+".json", jm, 0644))
		//os.Exit(0) // prototyping
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
