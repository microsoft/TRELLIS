import requests

def get_huggingface_model(model_name):
    # https://huggingface.co/JeffreyXiang/TRELLIS-image-large/resolve/main/pipeline.json
    url = f"https://huggingface.co/{model_name}/resolve/main/pipeline.json"

    try:
        print("url:", url)

        proxies = {
            "http": "http://127.0.0.1:52260",
            "https": "http://127.0.0.1:52260",
        }

        headers = {
            "Host": "huggingface.co", 
            "User-Agent":"unknown/None; hf_hub/0.29.1; python/3.10.16; torch/2.5.1+cu124",
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "Connection": "keep-alive"}
        response = requests.head(url=url, timeout=99999, proxies=proxies,allow_redirects=True, headers=headers, verify=False)
        if response.status_code == 200:
            print(f"Model {model_name} exists on Hugging Face")
        else:
            print(response.status_code)
    except Exception as e:
        print(e)



if __name__ == '__main__':
    get_huggingface_model("JeffreyXiang/TRELLIS-image-large")