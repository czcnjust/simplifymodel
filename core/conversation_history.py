"""
对话历史管理模块
保存所有用户和助手的对话记录
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class ConversationHistory:
    """对话历史管理器 - 简化版，只保存对话记录"""
    def __init__(self, history_file: str = "data/conversation_history.json"):
        """
        初始化对话历史管理器
        Args:
            history_file: 历史文件路径
        """
        self.history_file = history_file
        self.history: List[Dict] = []
        
        # 确保目录存在
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        # 加载历史记录
        self._load_history()
    
    def _load_history(self):
        """从文件加载历史记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                print(f"[INFO] 已加载对话历史: {len(self.history)} 条消息")
            except Exception as e:
                print(f"[WARNING] 加载对话历史失败: {e}，使用空历史")
                self.history = []
    
    def _save_history(self):
        """保存历史记录到文件"""
        try:
            data = {
                'history': self.history,
                'last_updated': datetime.now().isoformat(),
                'total_messages': len(self.history)
            }
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] 保存对话历史失败: {e}")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> Dict:
        """
        添加一条消息到历史记录
        
        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            metadata: 额外元数据（可以包含 model_code 等）
            
        Returns:
            添加的消息对象
        """
        message = {
            'id': f"msg_{len(self.history) + 1}",
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.history.append(message)
        self._save_history()
        
        return message
    
    def get_recent_messages(self, limit: int = 50) -> List[Dict]:
        """
        获取最近的消息
        
        Args:
            limit: 返回消息数量
            
        Returns:
            消息列表，格式为 [{"role": "user/assistant", "content": "..."}]
        """
        recent = self.history[-limit:]
        # 转换为 LLM API 需要的格式
        return [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in recent
        ]
    
    def get_full_history(self) -> List[Dict]:
        """
        获取完整的对话历史
        
        Returns:
            完整的消息列表
        """
        return self.history
    
    def get_history_as_text(self, limit: int = 100) -> str:
        """
        将对话历史转换为文本格式（用于给大模型）
        
        Args:
            limit: 最多包含的消息数
            
        Returns:
            格式化的对话文本
        """
        messages = self.history[-limit:]
        text_parts = []
        
        for msg in messages:
            role = "用户" if msg['role'] == 'user' else "助手"
            text_parts.append(f"[{role}] {msg['content']}")
        
        return "\n\n".join(text_parts)
    
    def clear_history(self):
        """清空历史记录"""
        self.history = []
        self._save_history()
        print("[INFO] 已清空对话历史")
    
    def get_message_count(self) -> int:
        """获取消息总数"""
        return len(self.history)


# 全局实例
conversation_history = ConversationHistory()
