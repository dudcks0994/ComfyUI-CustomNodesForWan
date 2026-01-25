import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "MyCustomNodes.BatchImageSave",
    async nodeCreated(node) {
        if (node.comfyClass !== "BatchImageSave") return;

        // Helper function to set folder path
        const setFolderPath = (path) => {
            const folderPathWidget = node.widgets.find(w => w.name === "folder_path");
            if (folderPathWidget) {
                folderPathWidget.value = path;
                if (folderPathWidget.callback) {
                    folderPathWidget.callback(folderPathWidget.value);
                }
                // Force node to update
                node.setDirtyCanvas(true);
            }
        };

        // Helper function for browser fallback
        const showBrowserFallback = () => {
            const userPath = prompt(
                "서버 폴더 선택기를 사용할 수 없습니다.\n전체 폴더 경로를 직접 입력하세요:\n\n예: E:\\output\\images"
            );
            
            if (userPath !== null && userPath.trim() !== "") {
                setFolderPath(userPath.trim());
            }
        };

        // Add folder selection button
        node.addWidget(
            "button",
            "📁 저장 폴더 선택",
            null,
            async () => {
                try {
                    // Try to use server-side folder selection via API
                    const response = await fetch("/loadmask/select_folder", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        }
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        if (data.success && data.folder_path) {
                            setFolderPath(data.folder_path);
                            console.log(`[BatchImageSave] 폴더 선택됨: ${data.folder_path}`);
                        } else if (data.error) {
                            console.warn(`[BatchImageSave] 폴더 선택 실패: ${data.error}`);
                            // Don't show fallback for cancelled selection
                            if (data.error !== "No folder selected") {
                                showBrowserFallback();
                            }
                        }
                    } else {
                        console.warn(`[BatchImageSave] API 응답 오류: ${response.status}`);
                        showBrowserFallback();
                    }
                } catch (error) {
                    console.error("[BatchImageSave] Error selecting folder:", error);
                    showBrowserFallback();
                }
            }
        );
    }
});

app.registerExtension({
    name: "MyCustomNodes.LoadBatchImage",
    async nodeCreated(node) {
        if (node.comfyClass !== "LoadBatchImage") return;

        // Helper function to set folder path
        const setFolderPath = (path) => {
            const folderPathWidget = node.widgets.find(w => w.name === "folder_path");
            if (folderPathWidget) {
                folderPathWidget.value = path;
                if (folderPathWidget.callback) {
                    folderPathWidget.callback(folderPathWidget.value);
                }
                // Force node to update
                node.setDirtyCanvas(true);
            }
        };

        // Helper function for browser fallback
        const showBrowserFallback = () => {
            const userPath = prompt(
                "서버 폴더 선택기를 사용할 수 없습니다.\n전체 폴더 경로를 직접 입력하세요:\n\n예: E:\\output\\images"
            );
            
            if (userPath !== null && userPath.trim() !== "") {
                setFolderPath(userPath.trim());
            }
        };

        // Add folder selection button
        node.addWidget(
            "button",
            "📁 폴더 선택",
            null,
            async () => {
                try {
                    // Try to use server-side folder selection via API
                    const response = await fetch("/loadmask/select_folder", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        }
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        if (data.success && data.folder_path) {
                            setFolderPath(data.folder_path);
                            console.log(`[LoadBatchImage] 폴더 선택됨: ${data.folder_path}`);
                        } else if (data.error) {
                            console.warn(`[LoadBatchImage] 폴더 선택 실패: ${data.error}`);
                            // Don't show fallback for cancelled selection
                            if (data.error !== "No folder selected") {
                                showBrowserFallback();
                            }
                        }
                    } else {
                        console.warn(`[LoadBatchImage] API 응답 오류: ${response.status}`);
                        showBrowserFallback();
                    }
                } catch (error) {
                    console.error("[LoadBatchImage] Error selecting folder:", error);
                    showBrowserFallback();
                }
            }
        );
    }
});
