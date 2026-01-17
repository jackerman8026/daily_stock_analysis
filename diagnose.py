# -*- coding: utf-8 -*-
"""
环境诊断脚本 (diagnose.py)

用于检查系统配置、密钥状态和网络连接。
"""
import os
import sys
import logging
from config import get_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_secret(name: str, value: str, critical: bool = True):
    """检查单个密钥"""
    if value:
        masked = value[:4] + "***" + value[-4:] if len(value) > 8 else "***"
        logger.info(f"✅ {name}: 已配置 ({masked})")
        return True
    else:
        if critical:
            logger.error(f"❌ {name}: 未配置 (严重)")
        else:
            logger.warning(f"⚠️ {name}: 未配置 (可选)")
        return False

def diagnose():
    print("\n=== A股智能分析系统 环境诊断 ===\n")
    config = get_config()

    # 1. 基础配置
    print("1. 基础配置检查")
    stock_list_str = os.getenv('STOCK_LIST', '')
    if stock_list_str:
        stocks = [s.strip() for s in stock_list_str.split(',') if s.strip()]
        logger.info(f"✅ STOCK_LIST: 已读取 {len(stocks)} 只股票 ({', '.join(stocks[:3])}...)")
        if len(stocks) < 1:
             logger.error("❌ STOCK_LIST: 解析后列表为空")
    else:
        logger.error("❌ STOCK_LIST: 未配置")

    print("\n2. AI 模型配置 (双模型)")
    # Gemini
    check_secret("GEMINI_API_KEY", config.gemini_api_key, critical=True)

    # DeepSeek / OpenAI
    dual_enabled = os.getenv('ENABLE_DUAL_MODEL', 'false').lower() == 'true'
    logger.info(f"ℹ️ ENABLE_DUAL_MODEL: {dual_enabled}")

    if dual_enabled:
        check_secret("OPENAI_API_KEY", config.openai_api_key, critical=True)
        if not config.openai_api_key:
            logger.error("   -> 双模型已开启，但缺少 OPENAI_API_KEY，导致第二份报告无法生成！")
    else:
        logger.info("   -> 双模型未开启，只会生成一份报告。")

    print("\n3. 新闻搜索配置")
    tavily_ok = check_secret("TAVILY_API_KEYS",  ','.join(config.tavily_api_keys) if config.tavily_api_keys else None, critical=False)

    if not tavily_ok:
        logger.warning("   -> 缺少搜索 Key，报告将没有新闻舆情分析！")

    print("\n4. 通知配置")
    check_secret("EMAIL_SENDER", config.email_sender, critical=False)
    check_secret("EMAIL_PASSWORD", config.email_password, critical=False)
    check_secret("CUSTOM_WEBHOOK_URLS", ','.join(config.custom_webhook_urls) if config.custom_webhook_urls else None, critical=False)

    print("\n=== 诊断结束 ===")

if __name__ == "__main__":
    diagnose()
