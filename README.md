# Minicpm-o-2.6 后端服务


## 编译 auto_gptq

```bash
cd minicpm_AutoGPTQ
python3.9 setup.py build
mv build/lib.linux-x86_64-3.9/* ../
```



## 修改模型文件

```bash
cp modeling_minicpmo.py "<MiniCPM-o-2_6-int4 path>"
```


## 测试

```bash
python3.9 ochat.py "<image-path>"
```



## API server

使用 [go-infer](https://github.com/jack139/go-infer)

```bash
python3.9 dispatcher.py 0
```



## 模型相关链接

https://github.com/OpenBMB/MiniCPM-o

https://github.com/OpenBMB/AutoGPTQ

https://huggingface.co/openbmb/MiniCPM-o-2_6-int4
