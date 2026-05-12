from .ImageResizeForWan import ResizeImageForWan, WanI2VResizeImage
from .ShowUtils import ShowInt, ShowFloat, ShowStringText
from .MathNode import MathNode
from .maskNode import LoadMaskNode, SaveMaskNode, MaskPaddingNode, MaskResizingNode, MyBlockifyMask, MyGrowMask, MyMaskSubtractNode
from .CalculateResolution import CalculateResolution
from .ImageBatchNode import BatchImageSave, LoadBatchImage
from .UtilNode import LoadMaskImageFromPath
from .WanAnimateNative import WanAnimateToVideoNative
from aiohttp import web
from server import PromptServer
from .ClipEncodeNode import CustomCLIPTextEncodeNode
from .CustomTextEncodeQwenImageEditNode import CustomTextEncodeQwenImageEditNode, CustomImagesEncodeNode

NODE_CLASS_MAPPINGS = {
    "WanI2VResizeImage": WanI2VResizeImage,
    "ResizeImageForWan": ResizeImageForWan,
    "ShowInt": ShowInt,
    "ShowFloat": ShowFloat,
    "ShowStringText": ShowStringText,
    "MathNode": MathNode,
    "LoadMaskNode": LoadMaskNode,
    "SaveMaskNode": SaveMaskNode,
    "MaskPaddingNode": MaskPaddingNode,
    "MaskResizingNode": MaskResizingNode,
    "CalculateResolution": CalculateResolution,
    "BatchImageSave": BatchImageSave,
    "LoadBatchImage": LoadBatchImage,
    "LoadMaskImageFromPath": LoadMaskImageFromPath,
    "MyBlockifyMask": MyBlockifyMask,
    "MyGrowMask": MyGrowMask,
    "WanAnimateToVideoNative": WanAnimateToVideoNative,
    "CustomCLIPTextEncodeNode": CustomCLIPTextEncodeNode,
    "MyMaskSubtractNode": MyMaskSubtractNode,
    "CustomTextEncodeQwenImageEditNode": CustomTextEncodeQwenImageEditNode,
    "CustomImagesEncodeNode": CustomImagesEncodeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanI2VResizeImage": "Wan I2V Resize Image",
    "ResizeImageForWan": "Wan I2V Resize Image (legacy)",
    "ShowInt": "ShowInt",
    "ShowFloat": "ShowFloat",
    "ShowStringText": "ShowStringText",
    "MathNode": "MathNode",
    "LoadMaskNode": "LoadMaskNode",
    "SaveMaskNode": "SaveMaskNode",
    "MaskPaddingNode": "MaskPaddingNode",
    "MaskResizingNode": "MaskResizingNode",
    "CalculateResolution": "CalculateResolution",
    "BatchImageSave": "BatchImageSave",
    "LoadBatchImage": "LoadBatchImage",
    "LoadMaskImageFromPath": "LoadMaskImageFromPath",
    "MyBlockifyMask": "MyBlockifyMask",
    "MyGrowMask": "MyGrowMask",
    "WanAnimateToVideoNative": "WanAnimateToVideoNative",
    "CustomCLIPTextEncodeNode": "CustomCLIPTextEncodeNode",
    "MyMaskSubtractNode": "MyMaskSubtractNode",
    "CustomTextEncodeQwenImageEditNode": "CustomTextEncodeQwenImageEditNode",
    "CustomImagesEncodeNode": "CustomImagesEncodeNode"
}
    
WEB_DIRECTORY = "./web"


# Server API for folder selection - register route on module load
@PromptServer.instance.routes.post("/loadmask/select_folder")
async def select_folder(request):
    try:
        import tkinter as tk
        from tkinter import filedialog
        import sys
        
        # Fix DPI scaling issue on Windows (makes text crisp on high-DPI displays)
        if sys.platform == 'win32':
            try:
                import ctypes
                # Enable DPI awareness for crisp rendering
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()  # Fallback for older Windows
                except Exception:
                    pass
        
        # Create a root window and hide it
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Open folder dialog
        folder_path = filedialog.askdirectory(title="마스크 폴더 선택")
        root.destroy()
        
        if folder_path:
            return web.json_response({"success": True, "folder_path": folder_path})
        else:
            return web.json_response({"success": False, "error": "No folder selected"})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})


import os

def strip_path(path):
    """경로에서 앞뒤 공백과 따옴표만 제거 (슬래시는 유지)"""
    path = path.strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path

def is_safe_path(path):
    """안전한 경로인지 확인 - 기본적으로 모든 경로 허용"""
    # 보안이 필요하면 환경변수로 제한 가능
    # if "MYCUSTOM_STRICT_PATHS" in os.environ:
    #     basedir = os.path.abspath('.')
    #     try:
    #         common_path = os.path.commonpath([basedir, path])
    #     except:
    #         return False
    #     return common_path == basedir
    return True

@PromptServer.instance.routes.get("/mycustom/getpath")
async def get_path(request):
    """경로를 받아서 해당 경로의 파일/폴더 목록을 반환"""
    query = request.rel_url.query
    if "path" not in query:
        return web.Response(status=204)
    
    path = os.path.abspath(strip_path(query["path"]))
    
    if not os.path.exists(path) or not is_safe_path(path):
        return web.json_response([])
    
    # extensions 파라미터 처리 (콤마로 구분된 확장자 목록)
    valid_extensions = query.get("extensions")
    if valid_extensions:
        valid_extensions = [ext.strip().lower() for ext in valid_extensions.split(",")]
    
    valid_items = []
    try:
        for item in os.scandir(path):
            try:
                if item.is_dir():
                    valid_items.append(item.name + "/")
                    continue
                if valid_extensions is None:
                    valid_items.append(item.name)
                else:
                    # 파일 확장자 체크
                    ext = item.name.split(".")[-1].lower() if "." in item.name else ""
                    if ext in valid_extensions:
                        valid_items.append(item.name)
            except OSError:
                # 깨진 심볼릭 링크 등 예외 처리
                pass
        
        # 수정 시간순 정렬 (에러 발생 시 0으로 처리)
        def get_mtime(f):
            try:
                return os.stat(os.path.join(path, f)).st_mtime
            except:
                return 0
        valid_items.sort(key=get_mtime)
    except PermissionError:
        return web.json_response([])
    
    return web.json_response(valid_items)


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
