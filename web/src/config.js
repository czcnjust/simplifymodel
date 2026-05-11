/**
 * 应用配置文件
 */

export const APP_CONFIG = {
  // Agent 欢迎消息
  welcomeMessage: `你好，我是神经网络优化 Agent。
你可以这样和我交互：
1. 粘贴 SimpleCNN 模型代码，将模型保存到sample.py;
2. 将上传的模型修改为包含残差结构，但是要适合量化；
3. 将当前的模型修改为包含注意力机制，但是要适合量化；
4. 训练sample.py；
5. 对sample.py进行ptq量化；
6. 对sample.py进行qat量化；
7. 对sample.py进行剪枝；
8. 用表格组织一下量化的知识。
9. 训练完成后，右侧会展示指标对比、模型结构图、生成代码和文件下载。
10. 结构图中可以点击每一层，查看该层的基本信息。`,
}
