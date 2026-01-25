import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// 경로에서 디렉토리 부분과 파일명을 분리
function path_stem(path) {
    // 윈도우와 유닉스 경로 모두 처리
    let i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
    if (i >= 0) {
        return [path.slice(0, i + 1), path.slice(i + 1)];
    }
    // 드라이브 레터만 있는 경우 (예: C:)
    if (path.length >= 2 && path[1] === ':') {
        return [path + "/", ""];
    }
    return ["", path];
}

// 텍스트가 너무 길면 줄임
function fitText(ctx, text, maxLength) {
    let fullLength = ctx.measureText(text).width;
    if (fullLength <= maxLength) {
        return [text, fullLength];
    }
    let len = Math.floor(maxLength / fullLength * text.length) - 3;
    if (len < 1) len = 1;
    let shortened = text.substring(0, len) + "...";
    return [shortened, ctx.measureText(shortened).width];
}

// 경로가 너무 길면 앞부분을 줄임
function fitPath(ctx, path, maxLength) {
    let fullLength = ctx.measureText(path).width;
    if (fullLength < maxLength) {
        return [path, fullLength];
    }
    let len = Math.floor(maxLength / fullLength * path.length) - 1;
    let displayPath = '';
    let filename = path_stem(path)[1];
    if (filename.length > len - 2) {
        displayPath = filename.substr(0, len);
    } else {
        let isAbs = path[0] == '/' || (path.length >= 2 && path[1] == ':');
        let pathPart = path.slice(isAbs ? 1 : 0, path.length - filename.length);
        let availLen = len - filename.length - (isAbs ? 4 : 3);
        displayPath = (isAbs ? path[0] : '') + "..." + pathPart.slice(-availLen) + filename;
    }
    return [displayPath, ctx.measureText(displayPath).width];
}

// 클릭 시 경로 검색 다이얼로그를 여는 함수
function searchBox(event, [x, y], node) {
    // 다이얼로그가 이미 열려있으면 무시
    if (this.prompt) return;
    this.prompt = true;

    let pathWidget = this;
    let dialog = document.createElement("div");
    dialog.className = "litegraph litesearchbox graphdialog rounded";
    dialog.innerHTML = '<span class="name">Path</span> <input autofocus="" type="text" class="value"><button class="rounded">OK</button><div class="helper"></div>';
    
    dialog.close = () => {
        dialog.remove();
        pathWidget.prompt = false;
    };
    
    document.body.append(dialog);
    
    if (app.canvas.ds.scale > 1) {
        dialog.style.transform = "scale(" + app.canvas.ds.scale + ")";
    }
    
    var name_element = dialog.querySelector(".name");
    var input = dialog.querySelector(".value");
    var options_element = dialog.querySelector(".helper");
    input.value = pathWidget.value;

    var timeout = null;
    let last_path = null;
    let extensions = pathWidget.options.mycustom_path_extensions;

    input.addEventListener("keydown", (e) => {
        dialog.is_modified = true;
        if (e.keyCode == 27) {
            // ESC
            dialog.close();
        } else if (e.keyCode == 13 && e.target.localName != "textarea") {
            // Enter
            pathWidget.value = input.value;
            if (pathWidget.callback) {
                pathWidget.callback(pathWidget.value);
            }
            dialog.close();
        } else {
            if (e.keyCode == 9) {
                // Tab - 첫 번째 옵션으로 자동완성
                if (options_element.firstChild) {
                    input.value = last_path + options_element.firstChild.innerText;
                }
                e.preventDefault();
                e.stopPropagation();
            } else if (e.ctrlKey && (e.keyCode == 87 || e.keyCode == 66)) {
                // Ctrl+w 또는 Ctrl+b - 상위 폴더로 이동
                input.value = path_stem(input.value.slice(0, -1))[0];
                e.preventDefault();
                e.stopPropagation();
            } else if (e.ctrlKey && e.keyCode == 71) {
                // Ctrl+g - 확장자 필터 임시 해제
                e.preventDefault();
                e.stopPropagation();
                extensions = undefined;
                last_path = null;
            }
            if (timeout) {
                clearTimeout(timeout);
            }
            timeout = setTimeout(updateOptions, 10);
            return;
        }
        this.prompt = false;
        e.preventDefault();
        e.stopPropagation();
    });

    var button = dialog.querySelector("button");
    button.addEventListener("click", (e) => {
        pathWidget.value = input.value;
        if (pathWidget.callback) {
            pathWidget.callback(pathWidget.value);
        }
        node.graph?.setDirtyCanvas(true);
        dialog.close();
        this.prompt = false;
    });

    var rect = app.canvas.canvas.getBoundingClientRect();
    var offsetx = -20;
    var offsety = -20;
    if (rect) {
        offsetx -= rect.left;
        offsety -= rect.top;
    }

    if (event) {
        dialog.style.left = event.clientX + offsetx + "px";
        dialog.style.top = event.clientY + offsety + "px";
    } else {
        dialog.style.left = app.canvas.canvas.width * 0.5 + offsetx + "px";
        dialog.style.top = app.canvas.canvas.height * 0.5 + offsety + "px";
    }

    // 검색 결과 표시 함수
    let options = [];
    
    function addResult(name, isDir) {
        let el = document.createElement("div");
        el.innerText = name;
        el.className = "litegraph lite-search-item";
        if (isDir) {
            el.className += " is-dir";
            el.style.fontWeight = "bold";
            el.style.color = "#8cf";
            el.addEventListener("click", (e) => {
                input.value = last_path + name;
                if (timeout) {
                    clearTimeout(timeout);
                }
                timeout = setTimeout(updateOptions, 10);
            });
        } else {
            el.addEventListener("click", (e) => {
                pathWidget.value = last_path + name;
                if (pathWidget.callback) {
                    pathWidget.callback(pathWidget.value);
                }
                dialog.close();
                pathWidget.prompt = false;
            });
        }
        options_element.appendChild(el);
    }

    async function updateOptions() {
        timeout = null;
        let [path, remainder] = path_stem(input.value);
        
        if (last_path != path) {
            // 서버에서 파일/폴더 목록 가져오기
            let params = { path: path };
            if (extensions) {
                params.extensions = extensions;
            }
            let optionsURL = api.apiURL('/mycustom/getpath?' + new URLSearchParams(params));
            try {
                let resp = await fetch(optionsURL);
                options = await resp.json();
                // 정렬을 위해 점을 null 문자로 대체
                options = options.map((o) => o.replace('.', '\0'));
                options = options.sort();
                options = options.map((o) => o.replace('\0', '.'));
            } catch (e) {
                console.error("[MyCustomNodes] Failed to fetch path:", e);
                options = [];
            }
            last_path = path;
        }
        
        options_element.innerHTML = '';
        
        // remainder로 필터링하여 표시
        for (let option of options) {
            if (option.toLowerCase().startsWith(remainder.toLowerCase())) {
                let isDir = option.endsWith('/');
                addResult(option, isDir);
            }
        }
    }

    setTimeout(async function() {
        input.focus();
        await updateOptions();
    }, 10);

    return dialog;
}

// 메인 확장 등록
app.registerExtension({
    name: "MyCustomNodes.FolderPathWidget",

    // 커스텀 위젯 타입 등록
    async getCustomWidgets() {
        return {
            MYCUSTOMPATH(node, inputName, inputData) {
                let w = {
                    name: inputName,
                    type: "MYCUSTOM.PATH",
                    value: "",
                    // 위젯 그리기
                    draw: function(ctx, node, widget_width, y, H) {
                        var show_text = app.canvas.ds.scale >= (app.canvas.low_quality_zoom_threshold ?? 0.5);
                        var margin = 15;
                        var text_color = LiteGraph.WIDGET_TEXT_COLOR;
                        var secondary_text_color = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
                        
                        ctx.textAlign = "left";
                        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
                        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
                        ctx.beginPath();
                        
                        if (show_text)
                            ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.5]);
                        else
                            ctx.rect(margin, y, widget_width - margin * 2, H);
                        ctx.fill();
                        
                        if (show_text) {
                            if (!this.disabled)
                                ctx.stroke();
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(margin, y, widget_width - margin * 2, H);
                            ctx.clip();

                            let freeWidth = widget_width - (margin * 2 + 40);
                            ctx.fillStyle = secondary_text_color;
                            const label = this.label || this.name;
                            if (label != null) {
                                let [labelDisplay, labelWidth] = fitText(ctx, label, freeWidth);
                                freeWidth -= labelWidth;
                                ctx.fillText(labelDisplay, margin * 2, y + H * 0.7);
                            }
                            ctx.fillStyle = this.value ? text_color : '#777';
                            ctx.textAlign = "right";
                            let disp_text = fitPath(ctx, String(this.value || this.options.placeholder || "클릭하여 경로 선택..."), freeWidth)[0];
                            ctx.fillText(disp_text, widget_width - margin * 2, y + H * 0.7);
                            ctx.restore();
                        }
                    },
                    // 클릭 시 searchBox 호출
                    mouse: searchBox,
                    options: {},
                };
                
                if (inputData.length > 1) {
                    w.options = inputData[1];
                    if (inputData[1].default) {
                        w.value = inputData[1].default;
                    }
                }

                if (!node.widgets) {
                    node.widgets = [];
                }
                node.widgets.push(w);
                return w;
            }
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // mycustom_path_extensions 옵션이 있는 노드 찾기
        const inputs = nodeData?.input;
        if (!inputs) return;

        const allInputs = { ...inputs.required, ...inputs.optional };
        let hasPathWidget = false;

        for (const [name, config] of Object.entries(allInputs)) {
            if (config[0] === "STRING" && config[1]?.mycustom_path_extensions !== undefined) {
                hasPathWidget = true;
                break;
            }
        }

        if (!hasPathWidget) return;

        // onNodeCreated 후킹 - STRING 위젯을 MYCUSTOMPATH 위젯으로 교체
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }

            let new_widgets = [];
            if (this.widgets) {
                for (let w of this.widgets) {
                    let config = allInputs[w.name];
                    if (!config) {
                        new_widgets.push(w);
                        continue;
                    }
                    if (w?.type == "text" && config[1]?.mycustom_path_extensions !== undefined) {
                        // STRING 위젯을 MYCUSTOMPATH 위젯으로 교체
                        new_widgets.push(app.widgets.MYCUSTOMPATH({}, w.name, ["MYCUSTOMPATH", config[1]]));
                    } else {
                        new_widgets.push(w);
                    }
                }
                this.widgets = new_widgets;
            }
        };
    }
});

console.log("[MyCustomNodes] Folder path widget extension loaded");
