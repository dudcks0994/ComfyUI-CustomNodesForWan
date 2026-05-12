import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Styles for the text area
const textStyles = {
  readOnly: true,
  opacity: 1,
  padding: "4px",
  paddingLeft: "7px",
  border: "1px solid #ccc",
  borderRadius: "5px",
  backgroundColor: "#222",
  color: "Lime",
  fontFamily: "Arial, sans-serif",
  fontSize: "14px",
  lineHeight: "1.4",
  resize: "none",
  overflowY: "auto",
};

const stringTextColors = [
  "white",
  "black",
  "gray",
  "silver",
  "red",
  "orange",
  "yellow",
  "lime",
  "green",
  "cyan",
  "aqua",
  "blue",
  "navy",
  "purple",
  "magenta",
  "pink",
  "brown",
  "gold",
  "tomato",
  "coral",
  "salmon",
  "violet",
  "plum",
  "skyblue",
  "deepskyblue",
  "dodgerblue",
  "turquoise",
  "springgreen",
];

function normalizeTextColor(value) {
  if (Array.isArray(value)) {
    value = value[0];
  }
  return value || "aqua";
}

function getStringTextColor(node, color) {
  return normalizeTextColor(color ?? node.widgets?.find((w) => w.name === "text_color")?.value);
}

function applyTextColor(node, color) {
  color = normalizeTextColor(color);
  node.widgets
    ?.filter((w) => w.name === "text" && w.inputEl)
    .forEach((w) => {
      w.inputEl.style.color = color;
    });
  app.graph.setDirtyCanvas(true, false);
}

function addWidgetValueRestore(nodeType, populate, transformValues) {
  const VALUES = Symbol();
  const configure = nodeType.prototype.configure;
  nodeType.prototype.configure = function () {
    this[VALUES] = arguments[0]?.widgets_values;
    return configure?.apply(this, arguments);
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    onConfigure?.apply(this, arguments);
    const widgetsValues = this[VALUES];
    if (widgetsValues?.length) {
      requestAnimationFrame(() => {
        const hasConvertedInputWidget = +(widgetsValues.length > 1 && this.inputs?.[0]?.widget);
        let values = widgetsValues.slice(hasConvertedInputWidget);
        if (transformValues) {
          values = transformValues.call(this, values);
        }
        populate.call(this, values);
      });
    }
  };
}

app.registerExtension({
  name: "MyCustomNodes.ShowStringText",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "ShowStringText") {
      function populate(text, textColor) {
        if (!Array.isArray(text)) {
          console.warn("populate expects an array, got:", text);
          return;
        }
        const color = getStringTextColor(this, textColor);

        if (this.widgets) {
          const pos = this.widgets.findIndex((w) => w.name === "text");
          if (pos !== -1) {
            for (let i = pos; i < this.widgets.length; i++) {
              this.widgets[i].onRemove?.();
            }
            this.widgets.length = pos;
          }
        } else {
          this.widgets = [];
        }

        text.forEach((list) => {
          const existingWidget = this.widgets.find(
            (w) => w.name === "text" && w.value === list
          );
          if (!existingWidget) {
            const w = ComfyWidgets["STRING"](
              this,
              "text",
              ["STRING", { multiline: true }],
              app
            ).widget;
            w.inputEl.readOnly = true;
            Object.assign(w.inputEl.style, textStyles);
            w.inputEl.style.color = color;
            w.value = list;
          }
        });
        applyTextColor(this, color);

        requestAnimationFrame(() => {
          const sz = this.computeSize();
          if (sz[0] < this.size[0]) sz[0] = this.size[0];
          if (sz[1] < this.size[1]) sz[1] = this.size[1];
          this.onResize?.(sz);
          app.graph.setDirtyCanvas(true, false);
        });
      }

      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        requestAnimationFrame(() => {
          const node = this;
          const colorWidget = this.widgets?.find((w) => w.name === "text_color");
          if (!colorWidget || colorWidget._myCustomTextColorHooked) {
            return;
          }
          const callback = colorWidget.callback;
          colorWidget.callback = function () {
            callback?.apply(this, arguments);
            applyTextColor(node, arguments[0] ?? this.value);
          };
          colorWidget._myCustomTextColorHooked = true;
        });
      };

      // When the node is executed we will be sent the input text, display this in the widget
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        const initialWidth = this.size[0];
        onExecuted?.apply(this, arguments);
        populate.call(this, message.text, message.text_color);
        this.size[0] = Math.max(initialWidth, 200); // Ensure minimum width
        // this.setSize(this.size[0], this.size[1]);
      };

      addWidgetValueRestore(nodeType, populate, (values) => {
        return stringTextColors.includes(values[0]) ? values.slice(1) : values;
      });
    }
  },
});

app.registerExtension({
  name: "Bjornulf.ShowJson",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "Bjornulf_ShowJson") {
      function populate(text) {
        if (!Array.isArray(text)) {
          console.warn("populate expects an array, got:", text);
          return;
        }

        if (this.widgets) {
          const pos = this.widgets.findIndex((w) => w.name === "text");
          if (pos !== -1) {
            for (let i = pos; i < this.widgets.length; i++) {
              this.widgets[i].onRemove?.();
            }
            this.widgets.length = pos;
          }
        } else {
          this.widgets = [];
        }

        text.forEach((list) => {
          const existingWidget = this.widgets.find(
            (w) => w.name === "text" && w.value === list
          );
          if (!existingWidget) {
            const w = ComfyWidgets["STRING"](
              this,
              "text",
              ["STRING", { multiline: true }],
              app
            ).widget;
            w.inputEl.readOnly = true;
            Object.assign(w.inputEl.style, textStyles);

            // Determine color based on type
            let color = "pink";

            w.inputEl.style.color = color;
            w.value = list;
          }
        });

        requestAnimationFrame(() => {
          const sz = this.computeSize();
          if (sz[0] < this.size[0]) sz[0] = this.size[0];
          if (sz[1] < this.size[1]) sz[1] = this.size[1];
          this.onResize?.(sz);
          app.graph.setDirtyCanvas(true, false);
        });
      }

      // When the node is executed we will be sent the input text, display this in the widget
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        const initialWidth = this.size[0];
        onExecuted?.apply(this, arguments);
        populate.call(this, message.text);
        this.size[0] = Math.max(initialWidth, 200); // Ensure minimum width
        // this.setSize(this.size[0], this.size[1]);
      };

    //   const onConfigure = nodeType.prototype.onConfigure;
    //   nodeType.prototype.onConfigure = function () {
    //       onConfigure?.apply(this, arguments);
    //       if (this.widgets_values?.length) {
    //           populate.call(this, this.widgets_values);
    //       }
    //   };
    }
  },
});

app.registerExtension({
  name: "MyCustomNodes.ShowInt",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "ShowInt") {
      function populate(text) {
        if (!Array.isArray(text)) {
          console.warn("populate expects an array, got:", text);
          return;
        }

        if (this.widgets) {
          const pos = this.widgets.findIndex((w) => w.name === "text");
          if (pos !== -1) {
            for (let i = pos; i < this.widgets.length; i++) {
              this.widgets[i].onRemove?.();
            }
            this.widgets.length = pos;
          }
        } else {
          this.widgets = [];
        }

        text.forEach((list) => {
          const existingWidget = this.widgets.find(
            (w) => w.name === "text" && w.value === list
          );
          if (!existingWidget) {
            const w = ComfyWidgets["STRING"](
              this,
              "text",
              ["STRING", { multiline: true }],
              app
            ).widget;
            w.inputEl.readOnly = true;
            Object.assign(w.inputEl.style, textStyles);

            // Determine color based on type
            let color = "#0096FF";

            w.inputEl.style.color = color;
            w.value = list;
          }
        });

        requestAnimationFrame(() => {
          const sz = this.computeSize();
          if (sz[0] < this.size[0]) sz[0] = this.size[0];
          if (sz[1] < this.size[1]) sz[1] = this.size[1];
          this.onResize?.(sz);
          app.graph.setDirtyCanvas(true, false);
        });
      }

      // When the node is executed we will be sent the input text, display this in the widget
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        populate.call(this, message.text);
      };

      addWidgetValueRestore(nodeType, populate);
    }
  },
});

app.registerExtension({
  name: "MyCustomNodes.ShowFloat",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "ShowFloat") {
      function populate(text) {
        if (!Array.isArray(text)) {
          console.warn("populate expects an array, got:", text);
          return;
        }

        if (this.widgets) {
          const pos = this.widgets.findIndex((w) => w.name === "text");
          if (pos !== -1) {
            for (let i = pos; i < this.widgets.length; i++) {
              this.widgets[i].onRemove?.();
            }
            this.widgets.length = pos;
          }
        } else {
          this.widgets = [];
        }

        text.forEach((list) => {
          const existingWidget = this.widgets.find(
            (w) => w.name === "text" && w.value === list
          );
          if (!existingWidget) {
            const w = ComfyWidgets["STRING"](
              this,
              "text",
              ["STRING", { multiline: true }],
              app
            ).widget;
            w.inputEl.readOnly = true;
            Object.assign(w.inputEl.style, textStyles);

            // Determine color based on type
            let color = "orange";

            w.inputEl.style.color = color;
            w.value = list;
          }
        });

        requestAnimationFrame(() => {
          const sz = this.computeSize();
          if (sz[0] < this.size[0]) sz[0] = this.size[0];
          if (sz[1] < this.size[1]) sz[1] = this.size[1];
          this.onResize?.(sz);
          app.graph.setDirtyCanvas(true, false);
        });
      }

      // When the node is executed we will be sent the input text, display this in the widget
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        populate.call(this, message.text);
      };

      addWidgetValueRestore(nodeType, populate);
    }
  },
});

// app.registerExtension({
//     name: "Bjornulf.ShowJson",
//     async beforeRegisterNodeDef(nodeType, nodeData, app) {
//         if (nodeData.name === "Bjornulf_ShowJson") {
//             function populate(text) {
//                 if (!Array.isArray(text)) {
//                     console.warn("populate expects an array, got:", text);
//                     return;
//                 }

//                 if (this.widgets) {
//                     const pos = this.widgets.findIndex((w) => w.name === "text");
//                     if (pos !== -1) {
//                         for (let i = pos; i < this.widgets.length; i++) {
//                             this.widgets[i].onRemove?.();
//                         }
//                         this.widgets.length = pos;
//                     }
//                 } else {
//                     this.widgets = [];
//                 }

//                 text.forEach((list) => {
//                     const existingWidget = this.widgets.find(w => w.name === "text" && w.value === list);
//                     if (!existingWidget) {
//                         const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
//                         w.inputEl.readOnly = true;
//                         Object.assign(w.inputEl.style, textStyles);

//                         // Determine color based on type
//                         let color = 'Lime'; // Default color for strings
//                         const value = list.toString().trim();

//                         if (/^-?\d+$/.test(value)) {
//                             color = '#0096FF'; // Integer
//                         } else if (/^-?\d*\.?\d+$/.test(value)) {
//                             color = 'orange'; // Float
//                         } else if (value.startsWith("If-Else ERROR: ")) {
//                             color = 'red'; // If-Else ERROR lines
//                         } else if (value.startsWith("tensor(")) {
//                             color = '#0096FF'; // Lines starting with "tensor("
//                         }

//                         w.inputEl.style.color = color;
//                         w.value = list;
//                     }
//                 });

//                 requestAnimationFrame(() => {
//                     const sz = this.computeSize();
//                     if (sz[0] < this.size[0]) sz[0] = this.size[0];
//                     if (sz[1] < this.size[1]) sz[1] = this.size[1];
//                     this.onResize?.(sz);
//                     app.graph.setDirtyCanvas(true, false);
//                 });
//             }

//             // When the node is executed we will be sent the input text, display this in the widget
//             const onExecuted = nodeType.prototype.onExecuted;
//             nodeType.prototype.onExecuted = function (message) {
//                 onExecuted?.apply(this, arguments);
//                 populate.call(this, message.text);
//             };

//             const onConfigure = nodeType.prototype.onConfigure;
//             nodeType.prototype.onConfigure = function () {
//                 onConfigure?.apply(this, arguments);
//                 if (this.widgets_values?.length) {
//                     populate.call(this, this.widgets_values);
//                 }
//             };
//         }
//     },
// });
