package minicpmo

import (
	"fmt"
	"log"
	"strconv"
	"encoding/base64"

	"github.com/jack139/go-infer/helper"
)



/*  定义模型相关参数和方法  */
type OCHAT struct{}

func (x *OCHAT) Init() error {
	return nil
}

func (x *OCHAT) ApiPath() string {
	return "/api/ocr/rec"
}

func (x *OCHAT) CustomQueue() string {
	return helper.Settings.Customer["OCR_QUEUE"]
}

func (x *OCHAT) ApiEntry(reqData *map[string]interface{}) (*map[string]interface{}, error) {
	log.Println("Api_OCHAT")

	// 检查参数
	imageBase64, ok := (*reqData)["image"].(string)
	if !ok {
		return &map[string]interface{}{"code":9101}, fmt.Errorf("need image")
	}

	text, ok := (*reqData)["text"].(string)
	if !ok {
		return &map[string]interface{}{"code":9102}, fmt.Errorf("need text")
	}

	// 解码base64
	image, err  := base64.StdEncoding.DecodeString(imageBase64)
	if err!=nil {
		return &map[string]interface{}{"code":9901}, err
	}

	// 检查图片大小
	maxSize, _ := strconv.Atoi(helper.Settings.Customer["OCR_MAX_IMAGE_SIZE"])
	if len(image) > maxSize {
		return &map[string]interface{}{"code":9002}, fmt.Errorf("图片数据太大")
	}

	// 构建请求参数
	reqDataMap := map[string]interface{}{
		"image": imageBase64,
		"text": text,
	}

	return &reqDataMap, nil
}


// OCHAT 推理 - 不在这里实现，由 python dispatcher 实现
func (x *OCHAT) Infer(reqId string, reqData *map[string]interface{}) (*map[string]interface{}, error) {
	log.Println("Infer_OCHAT - Do nothing")

	return &map[string]interface{}{}, nil
}
