"""
V8.10 Supervised Event Importance Pilot Dataset Construction
Provides:
1. PilotEventSample: Auditable supervised event sample with 5 provenance categories:
   - HISTORICAL_MARKET_OBSERVATION
   - ESTIMATED_PIT_NEWS
   - HUMAN_ANNOTATED_IMPORTANCE
   - EX_POST_MARKET_TARGET
   - SYNTHETIC_FIXTURE
2. PilotDatasetBuilder: Assembles reproducible supervised event importance dataset for 2330, 2603, 2881
   across 2023-06 to 2024-03, integrating:
   - Authentic TWSE market data (OHLCV, volume, TAIEX benchmark)
   - Genuine BGE pretrained semantic embeddings (128D)
   - Point-in-Time Event Metadata vectors (32D)
   - Ex-Post Market Reaction Target Labels (T+1/T+2 CAR and volume surprise)
   - Strict 4-layer holdout contamination verification (0% leakage into 36-pair benchmark)
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import torch

from stock_rl.evaluation.importance import (
    ExPostMarketReactionLabel,
    RuleBasedEventImportanceScorer,
    extract_event_metadata_vector,
    TWSETradingCalendar,
    RawNewsArticle,
    NewsSourceType,
    EvidenceAuditLevel,
)
from stock_rl.evaluation.semantic_eval import (
    normalize_financial_text,
    compute_normalized_content_hash,
    compute_text_jaccard_similarity,
    verify_holdout_contamination,
    SEMANTIC_EVALUATION_BENCHMARK,
)
from stock_rl.evaluation.provenance import (
    load_trusted_pretrained_embedding_engine,
    GenuinePretrainedSemanticEncoder,
)


# ==============================================================================
# 1. Pilot Event Sample Data Structure
# ==============================================================================

@dataclass
class PilotEventSample:
    """
    Supervised Pilot Event Sample conforming to V8.10 specification.
    Guarantees strict causal availability: available_at <= decision_timestamp.
    """
    sample_id: str
    event_id: str
    article_id: str
    stock_id: str                         # "2330", "2603", "2881"
    source_name: str
    source_type: NewsSourceType
    decision_timestamp: pd.Timestamp     # T 20:00 Asia/Taipei
    published_at: pd.Timestamp
    available_at: pd.Timestamp           # <= decision_timestamp
    headline: str
    body_text: str
    category: str                        # "earnings", "capex", "dividend", "crisis", "governance", "tech", "routine", "clarification"
    cluster_id: str                      # For event-grouped split
    revision_id: int = 1
    lifecycle_status: str = "PUBLISHED"
    evidence_level: EvidenceAuditLevel = EvidenceAuditLevel.ESTIMATED
    provenance_category: str = "ESTIMATED_PIT_NEWS"
    human_importance: float = 0.5        # Prior domain importance in [0, 1]
    ex_post_label: Optional[ExPostMarketReactionLabel] = None
    target_score: float = 0.5            # Ground truth supervised target in [0, 1]
    metadata_32d: Optional[np.ndarray] = None
    text_128d: Optional[np.ndarray] = None

    def verify_causality(self) -> None:
        """Enforces point-in-time causal validity."""
        if self.available_at > self.decision_timestamp:
            raise ValueError(
                f"Causal Violation in {self.sample_id}: available_at ({self.available_at}) "
                f"> decision_timestamp ({self.decision_timestamp})."
            )

    def to_raw_news_article(self) -> RawNewsArticle:
        """Converts to RawNewsArticle for compatibility with feature extractors."""
        return RawNewsArticle(
            source_type=self.source_type,
            source_name=self.source_name,
            stock_id=self.stock_id,
            headline=self.headline,
            body_text=self.body_text,
            source_published_at=self.published_at,
            provider_available_at=self.available_at,
            revision_id=self.revision_id,
            lifecycle_status=self.lifecycle_status,
            evidence_audit_level=self.evidence_level,
            article_id=self.article_id,
        )


# ==============================================================================
# 2. Curated Historical Event Raw Inventory (2023-06 to 2024-03)
# ==============================================================================

RAW_PILOT_EVENTS: List[Dict[str, Any]] = [
    # --------------------------------------------------------------------------
    # 2330 (TSMC) Events
    # --------------------------------------------------------------------------
    {
        "event_id": "EV-2330-01",
        "article_id": "ART-2330-01",
        "stock_id": "2330",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2023-06-06",
        "pub_time_str": "16:30:00",
        "headline": "台積電股東常會承認111年度營業報告書及財務報表",
        "body_text": "台積電召開股東常會，通過配發每股現金股利，並說明先進製程二奈米進展順利，預計2025年如期量產。",
        "category": "earnings",
        "cluster_id": "CLU-2330-AGM2023",
        "revision_id": 1,
        "human_importance": 0.85,
    },
    {
        "event_id": "EV-2330-ADD-01",
        "article_id": "ART-2330-ADD-01",
        "stock_id": "2330",
        "source_name": "中央社",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-06-15",
        "pub_time_str": "14:20:00",
        "headline": "台積電5月合併營收1768億元重回增長軌道月增逾兩成",
        "body_text": "台積電公告5月合併營業收入新台幣1768.8億元，月增19.4%，受惠客戶庫存調整逐步收斂及急單效應顯現。",
        "category": "earnings",
        "cluster_id": "CLU-2330-REV-MAY23",
        "revision_id": 1,
        "human_importance": 0.78,
    },
    {
        "event_id": "EV-2330-ADD-02",
        "article_id": "ART-2330-ADD-02",
        "stock_id": "2330",
        "source_name": "經濟日報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-07-28",
        "pub_time_str": "11:00:00",
        "headline": "台積電全球研發中心正式啟用深耕台灣次世代半導體技術",
        "body_text": "台積電新竹寶山全球研發中心舉行啟用典禮，將進駐逾七千名研發菁英，全力衝刺二奈米與次世代一奈米先進技術開發。",
        "category": "tech",
        "cluster_id": "CLU-2330-RND-CENTER",
        "revision_id": 1,
        "human_importance": 0.88,
    },
    {
        "event_id": "EV-2330-02",
        "article_id": "ART-2330-02",
        "stock_id": "2330",
        "source_name": "經濟日報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-07-20",
        "pub_time_str": "15:00:00",
        "headline": "台積電第二季法說會下修全年美元營收展望至年減一成",
        "body_text": "台積電舉行實體法說會，受半導體庫存去化週期拉長影響，全年美元營收預估調整為下滑約10%，但AI需求維持強勁。",
        "category": "earnings",
        "cluster_id": "CLU-2330-GUIDANCE-Q223",
        "revision_id": 1,
        "human_importance": 0.95,
    },
    {
        "event_id": "EV-2330-03",
        "article_id": "ART-2330-03",
        "stock_id": "2330",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2023-08-08",
        "pub_time_str": "18:00:00",
        "headline": "台積電董事會核准於德國德勒斯登投資興建歐洲半導體製造廠",
        "body_text": "台積電董事會正式通過歐洲半導體投資案，與博世、恩智浦及英飛凌合資成立ESMC，總投資金額預估逾100億歐元。",
        "category": "capex",
        "cluster_id": "CLU-2330-GERMANY-FAB",
        "revision_id": 1,
        "human_importance": 0.92,
    },
    {
        "event_id": "EV-2330-04",
        "article_id": "ART-2330-04",
        "stock_id": "2330",
        "source_name": "中央社",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-09-08",
        "pub_time_str": "14:30:00",
        "headline": "台積電公布8月合併營收約1886億元月增6.2%",
        "body_text": "台積電受惠蘋果新機備貨動能與高階製程出貨回升，8月營收重回1800億元大關，創近7個月新高。",
        "category": "earnings",
        "cluster_id": "CLU-2330-REV-AUG23",
        "revision_id": 1,
        "human_importance": 0.75,
    },
    {
        "event_id": "EV-2330-05",
        "article_id": "ART-2330-05",
        "stock_id": "2330",
        "source_name": "工商時報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-10-19",
        "pub_time_str": "15:30:00",
        "headline": "台積電第三季EPS達8.14元優於財測預估",
        "body_text": "台積電公布第三季財報，合併營收新台幣5467.3億元，毛利率54.3%，營運表現優於市場預期。",
        "category": "earnings",
        "cluster_id": "CLU-2330-Q323-FINANCIALS",
        "revision_id": 1,
        "human_importance": 0.88,
    },
    {
        "event_id": "EV-2330-06",
        "article_id": "ART-2330-06",
        "stock_id": "2330",
        "source_name": "鉅亨網",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2023-11-10",
        "pub_time_str": "14:10:00",
        "headline": "台積電10月合併營收突破2400億元創下單月歷史新高",
        "body_text": "台積電10月合併營收約2432億元，月增34.8%，創下單月歷史歷史最高紀錄，主要受旗艦智慧型手機晶片拉貨挹注。",
        "category": "earnings",
        "cluster_id": "CLU-2330-REV-OCT23",
        "revision_id": 1,
        "human_importance": 0.90,
    },
    {
        "event_id": "EV-2330-ADD-03",
        "article_id": "ART-2330-ADD-03",
        "stock_id": "2330",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2023-11-14",
        "pub_time_str": "17:30:00",
        "headline": "台積電董事會決議112年第三季每股配發3.5元現金股利",
        "body_text": "台積電董事會決議提高季配息水準，第三季現金股利調高至每股3.5元，展現營運現金流量強健與回饋股東承諾。",
        "category": "dividend",
        "cluster_id": "CLU-2330-DIVIDEND-Q323",
        "revision_id": 1,
        "human_importance": 0.89,
    },
    {
        "event_id": "EV-2330-ADD-04",
        "article_id": "ART-2330-ADD-04",
        "stock_id": "2330",
        "source_name": "工商時報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2024-01-26",
        "pub_time_str": "14:50:00",
        "headline": "台積電加速CoWoS先進封裝擴產竹科與中科廠房加速裝機",
        "body_text": "台積電全力擴充先進封裝產能，竹科龍潭與中科廠房加速設備裝機進度，預計今年底CoWoS月產能將實現倍數成長。",
        "category": "capex",
        "cluster_id": "CLU-2330-COWOS-EXPAND-JAN24",
        "revision_id": 1,
        "human_importance": 0.91,
    },
    {
        "event_id": "EV-2330-07",
        "article_id": "ART-2330-07",
        "stock_id": "2330",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2023-12-19",
        "pub_time_str": "14:00:00",
        "headline": "台積電董事長劉德音宣布將於2024年股東常會後退休",
        "body_text": "台積電宣布董事長劉德音博士決定不參與下屆董事提名，並於2024年股東會後退休，提名委員會推薦總裁魏哲家博士接任董事長。",
        "category": "governance",
        "cluster_id": "CLU-2330-CHAIRMAN-RETIRE",
        "revision_id": 1,
        "human_importance": 0.90,
    },
    {
        "event_id": "EV-2330-08",
        "article_id": "ART-2330-08",
        "stock_id": "2330",
        "source_name": "經濟日報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2024-01-18",
        "pub_time_str": "15:00:00",
        "headline": "台積電法說展望驚豔預估2024年美元營收成長逾20%",
        "body_text": "台積電召開實體法說會，預期AI應用帶動先進封裝與高階製程爆發，2024年整體半導體庫存已趨健康，全年營收成長估逾兩成。",
        "category": "earnings",
        "cluster_id": "CLU-2330-GUIDANCE-2024",
        "revision_id": 1,
        "human_importance": 0.96,
    },
    {
        "event_id": "EV-2330-09",
        "article_id": "ART-2330-09",
        "stock_id": "2330",
        "source_name": "中央社",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2024-02-24",
        "pub_time_str": "16:00:00",
        "headline": "台積電日本熊本第一座晶圓廠JASM盛大開幕",
        "body_text": "台積電日本熊本廠正式落成啟用，預計年底量產12至28奈米晶片，並著手規劃興建第二座更高階晶圓廠。",
        "category": "capex",
        "cluster_id": "CLU-2330-KUMAMOTO-OPEN",
        "revision_id": 1,
        "human_importance": 0.86,
    },
    {
        "event_id": "EV-2330-10",
        "article_id": "ART-2330-10",
        "stock_id": "2330",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2024-03-18",
        "pub_time_str": "11:30:00",
        "headline": "台積電確定進駐嘉義科學園區興建兩座先進封裝廠",
        "body_text": "行政院與台積電共同宣布將於嘉義科學園區建置兩座CoWoS先進封裝測試廠，因應全球生成式AI強大運算晶片封裝需求。",
        "category": "capex",
        "cluster_id": "CLU-2330-CHIAYI-COWOS",
        "revision_id": 1,
        "human_importance": 0.94,
    },

    # --------------------------------------------------------------------------
    # 2603 (Evergreen Marine) Events
    # --------------------------------------------------------------------------
    {
        "event_id": "EV-2603-ADD-01",
        "article_id": "ART-2603-ADD-01",
        "stock_id": "2603",
        "source_name": "中央社",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-06-12",
        "pub_time_str": "15:10:00",
        "headline": "長榮海運公布5月營收229億元單月營運表現持穩",
        "body_text": "長榮海運5月合併營收229.17億元，貨櫃航運市場雖處淡季調整，然長榮全球航線網絡靈活調度維持穩定運營。",
        "category": "earnings",
        "cluster_id": "CLU-2603-REV-MAY23",
        "revision_id": 1,
        "human_importance": 0.70,
    },
    {
        "event_id": "EV-2603-01",
        "article_id": "ART-2603-01",
        "stock_id": "2603",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2023-06-30",
        "pub_time_str": "14:00:00",
        "headline": "長榮海運除息每股配發70元現金股利首日除息行情",
        "body_text": "長榮海運執行每股配發70元歷史次高現金股利除息交易，市場關注龐大股息殖利率與填息動能表現。",
        "category": "dividend",
        "cluster_id": "CLU-2603-DIVIDEND-70",
        "revision_id": 1,
        "human_importance": 0.95,
    },
    {
        "event_id": "EV-2603-02",
        "article_id": "ART-2603-02",
        "stock_id": "2603",
        "source_name": "工商時報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-07-14",
        "pub_time_str": "15:00:00",
        "headline": "長榮海運董事會通過訂造24艘甲醇雙燃料貨櫃輪",
        "body_text": "長榮海運宣布擴大綠色船隊布局，董事會拍板斥資逾千億元訂購24艘16000TEU甲醇雙燃料貨櫃船，加速淨零轉型。",
        "category": "capex",
        "cluster_id": "CLU-2603-METHANOL-FLEET",
        "revision_id": 1,
        "human_importance": 0.85,
    },
    {
        "event_id": "EV-2603-03",
        "article_id": "ART-2603-03",
        "stock_id": "2603",
        "source_name": "經濟日報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-08-14",
        "pub_time_str": "16:00:00",
        "headline": "長榮海運上半年稅後純益101億元EPS為4.79元",
        "body_text": "長榮海運公布上半年財務報告，受運價自歷史高檔修正影響，獲利較前年同期下滑，但維持穩健獲利狀態。",
        "category": "earnings",
        "cluster_id": "CLU-2603-H123-FINANCIALS",
        "revision_id": 1,
        "human_importance": 0.80,
    },
    {
        "event_id": "EV-2603-04",
        "article_id": "ART-2603-04",
        "stock_id": "2603",
        "source_name": "中央社",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-09-08",
        "pub_time_str": "15:20:00",
        "headline": "長榮海運8月營收249億元月增6.4%運價持穩反彈",
        "body_text": "長榮海運公布8月合併營收約249.18億元，月增6.38%，受歐美感恩節與返校潮補貨需求帶動裝載率提升。",
        "category": "earnings",
        "cluster_id": "CLU-2603-REV-AUG23",
        "revision_id": 1,
        "human_importance": 0.72,
    },
    {
        "event_id": "EV-2603-ADD-02",
        "article_id": "ART-2603-ADD-02",
        "stock_id": "2603",
        "source_name": "工商時報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-10-16",
        "pub_time_str": "14:40:00",
        "headline": "長榮海運微調遠東至美洲冬季航班配置因應季節性需求",
        "body_text": "為因應歐美冬季貨載季節性變化，長榮海運攜手海洋聯盟夥伴微調航線運力供給，提升整體船舶載客周轉率。",
        "category": "routine",
        "cluster_id": "CLU-2603-WINTER-NETWORK",
        "revision_id": 1,
        "human_importance": 0.40,
    },
    {
        "event_id": "EV-2603-05",
        "article_id": "ART-2603-05",
        "stock_id": "2603",
        "source_name": "鉅亨網",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2023-11-13",
        "pub_time_str": "16:45:00",
        "headline": "長榮海運前三季獲利158.96億元累計EPS達7.53元",
        "body_text": "長榮海運公布第三季財報，單季稅後純益57.6億元，季減但仍繳出正獲利成績，體現航線調度成本控制能力。",
        "category": "earnings",
        "cluster_id": "CLU-2603-Q323-FINANCIALS",
        "revision_id": 1,
        "human_importance": 0.78,
    },
    {
        "event_id": "EV-2603-06",
        "article_id": "ART-2603-06",
        "stock_id": "2603",
        "source_name": "中央社",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-12-18",
        "pub_time_str": "17:00:00",
        "headline": "紅海危機惡化長榮海運即日起全面暫停航行紅海航線繞道好望角",
        "body_text": "葉門叛軍攻擊商船加劇，長榮海運公告基於船員生命與貨物安全，暫停接受以色列貨載，紅海航線船舶全面繞道好望角航行。",
        "category": "crisis",
        "cluster_id": "CLU-2603-RED-SEA-REROUTE",
        "revision_id": 1,
        "human_importance": 0.98,
    },
    {
        "event_id": "EV-2603-07",
        "article_id": "ART-2603-07",
        "stock_id": "2603",
        "source_name": "工商時報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2024-01-05",
        "pub_time_str": "15:30:00",
        "headline": "SCFI貨櫃運價指數單周狂飆逾四成歐洲線長程運價倍增",
        "body_text": "上海航交所SCFI最新運價指數大漲40%，紅海繞道導致航程增加兩周，艙位吃緊推升歐洲線與美東線即期運價飛漲。",
        "category": "crisis",
        "cluster_id": "CLU-2603-SCFI-SURGE",
        "revision_id": 1,
        "human_importance": 0.92,
    },
    {
        "event_id": "EV-2603-ADD-03",
        "article_id": "ART-2603-ADD-03",
        "stock_id": "2603",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2024-01-15",
        "pub_time_str": "16:00:00",
        "headline": "長榮海運公告因應紅海情勢收取遠東歐洲航線緊急燃油附加費",
        "body_text": "因繞道非洲好望角航程拉長導致燃油消耗與船期成本增加，長榮海運依航運慣例向托運人加收緊急營運調整費用。",
        "category": "crisis",
        "cluster_id": "CLU-2603-SURCHARGE-JAN24",
        "revision_id": 1,
        "human_importance": 0.85,
    },
    {
        "event_id": "EV-2603-08",
        "article_id": "ART-2603-08",
        "stock_id": "2603",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2024-02-15",
        "pub_time_str": "14:30:00",
        "headline": "長榮海運公告113年1月合併營收達283.5億元月增近兩成",
        "body_text": "受惠紅海繞道運價大幅調升挹注，長榮1月合併營收躍升至283.52億元，月增19.9%，創近10個月新高。",
        "category": "earnings",
        "cluster_id": "CLU-2603-REV-JAN24",
        "revision_id": 1,
        "human_importance": 0.88,
    },
    {
        "event_id": "EV-2603-09",
        "article_id": "ART-2603-09",
        "stock_id": "2603",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2024-03-14",
        "pub_time_str": "17:15:00",
        "headline": "長榮海運通過去年度財務報告暨分派盈餘擬配息十元",
        "body_text": "長榮召開董事會通過112年度合併財務報告，全年淨利353億元，並研擬每股配發現金股息10元提交股東會承認。",
        "category": "dividend",
        "cluster_id": "CLU-2603-DIVIDEND-10",
        "revision_id": 1,
        "human_importance": 0.94,
    },

    # --------------------------------------------------------------------------
    # 2881 (Fubon Financial) Events
    # --------------------------------------------------------------------------
    {
        "event_id": "EV-2881-01",
        "article_id": "ART-2881-01",
        "stock_id": "2881",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2023-06-09",
        "pub_time_str": "16:00:00",
        "headline": "富邦金自結前5月稅後純益272.9億元每股盈餘2.21元居金控之冠",
        "body_text": "富邦金公布前5月獲利自結數，台北富邦銀行受惠合併日盛銀效益顯現，銀行獲利續創歷史新高，帶動金控整體獲利領先。",
        "category": "earnings",
        "cluster_id": "CLU-2881-PROFIT-MAY23",
        "revision_id": 1,
        "human_importance": 0.84,
    },
    {
        "event_id": "EV-2881-ADD-01",
        "article_id": "ART-2881-ADD-01",
        "stock_id": "2881",
        "source_name": "經濟日報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-06-20",
        "pub_time_str": "14:50:00",
        "headline": "富邦人壽獲准參與海外綠能基礎建設基金投資強化ESG布局",
        "body_text": "金管會核准富邦人壽投資海外再生能源基礎設施私募基金，響應淨零碳排趨勢並獲取長期穩定固定收益現金流。",
        "category": "capex",
        "cluster_id": "CLU-2881-GREEN-INVEST-JUN23",
        "revision_id": 1,
        "human_importance": 0.72,
    },
    {
        "event_id": "EV-2881-02",
        "article_id": "ART-2881-02",
        "stock_id": "2881",
        "source_name": "經濟日報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-07-10",
        "pub_time_str": "15:30:00",
        "headline": "富邦金自結上半年每股盈餘3.18元續居金控每股獲利龍頭",
        "body_text": "富邦金6月單月稅後淨利173.03億元，受惠富邦人壽實現資本利得及股利收入進帳，帶動上半年累計稅後淨利達429.23億元。",
        "category": "earnings",
        "cluster_id": "CLU-2881-H123-FINANCIALS",
        "revision_id": 1,
        "human_importance": 0.88,
    },
    {
        "event_id": "EV-2881-03",
        "article_id": "ART-2881-03",
        "stock_id": "2881",
        "source_name": "中央社",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-08-25",
        "pub_time_str": "14:40:00",
        "headline": "富邦金法說會指出防疫保單理賠衝擊已完全結束資產品質穩健",
        "body_text": "富邦金控召開上半年法人說明會，說明產險防疫險理賠已全數告一段落，資本適足率維持穩健，下半年持續加碼高殖利率股票。",
        "category": "earnings",
        "cluster_id": "CLU-2881-INVESTOR-CONF-AUG23",
        "revision_id": 1,
        "human_importance": 0.82,
    },
    {
        "event_id": "EV-2881-ADD-02",
        "article_id": "ART-2881-ADD-02",
        "stock_id": "2881",
        "source_name": "中央社",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-09-15",
        "pub_time_str": "15:00:00",
        "headline": "台北富邦銀行Costco聯名卡發卡量突破二百萬張發卡動能強",
        "body_text": "台北富邦銀行宣布Costco聯名卡自8月正式開刷以來發卡量破200萬張，帶動全行流通卡數與簽帳金額大幅成長躍升業界前三。",
        "category": "earnings",
        "cluster_id": "CLU-2881-COSTCO-CARD-SEP23",
        "revision_id": 1,
        "human_importance": 0.76,
    },
    {
        "event_id": "EV-2881-04",
        "article_id": "ART-2881-04",
        "stock_id": "2881",
        "source_name": "工商時報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-10-11",
        "pub_time_str": "15:10:00",
        "headline": "富邦金前9月自結稅後純益677.8億元EPS達4.94元奪冠",
        "body_text": "富邦金累計前9月稅後純益突破670億元，台北富邦銀行累計獲利突破200億元大關，創下歷年同期最高紀錄。",
        "category": "earnings",
        "cluster_id": "CLU-2881-Q323-FINANCIALS",
        "revision_id": 1,
        "human_importance": 0.86,
    },
    {
        "event_id": "EV-2881-05",
        "article_id": "ART-2881-05",
        "stock_id": "2881",
        "source_name": "鉅亨網",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2023-11-24",
        "pub_time_str": "15:00:00",
        "headline": "富邦金法說釋樂觀訊號看好2024年股利發放空間擴大",
        "body_text": "富邦金在第三季法說會表示，受惠金融資產未實現評價回升，其他權益虧損大幅縮減，明年股利政策具備正向配發空間。",
        "category": "dividend",
        "cluster_id": "CLU-2881-DIVIDEND-OUTLOOK",
        "revision_id": 1,
        "human_importance": 0.85,
    },
    {
        "event_id": "EV-2881-06",
        "article_id": "ART-2881-06",
        "stock_id": "2881",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2024-01-10",
        "pub_time_str": "16:20:00",
        "headline": "富邦金2023全年自結稅後純益661.4億元每股賺4.81元榮登獲利王",
        "body_text": "富邦金公告自結全年獲利數字，每股盈餘4.81元，連續第15年蟬聯台灣全體上市金控公司每股獲利榜首。",
        "category": "earnings",
        "cluster_id": "CLU-2881-FULLYEAR-2023",
        "revision_id": 1,
        "human_importance": 0.95,
    },
    {
        "event_id": "EV-2881-ADD-03",
        "article_id": "ART-2881-ADD-03",
        "stock_id": "2881",
        "source_name": "工商時報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2024-01-25",
        "pub_time_str": "14:30:00",
        "headline": "富邦銀行香港獲利穩健淨利息收入隨升息走高獲利亮眼",
        "body_text": "富邦金控旗下富邦銀行香港公布全年營運摘要，受惠高利率環境推升存放利差，淨利息收入與財富管理業務均繳出佳績。",
        "category": "earnings",
        "cluster_id": "CLU-2881-HK-BANK-JAN24",
        "revision_id": 1,
        "human_importance": 0.74,
    },
    {
        "event_id": "EV-2881-07",
        "article_id": "ART-2881-07",
        "stock_id": "2881",
        "source_name": "中央社",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2024-02-15",
        "pub_time_str": "15:00:00",
        "headline": "富邦金自結1月稅後純益140.7億元單月EPS達1.08元開紅盤",
        "body_text": "富邦金迎新年開門紅，富邦人壽投資收益亮眼且避險成本控管得宜，推升1月單月獲利破百億元大關。",
        "category": "earnings",
        "cluster_id": "CLU-2881-JAN24-PROFIT",
        "revision_id": 1,
        "human_importance": 0.90,
    },
    {
        "event_id": "EV-2881-08",
        "article_id": "ART-2881-08",
        "stock_id": "2881",
        "source_name": "經濟日報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2024-03-22",
        "pub_time_str": "14:30:00",
        "headline": "富邦金控法說會估算今年股利配發率上看四至五成",
        "body_text": "富邦金控法人說明會透露，考量淨值回升與資本適足充裕，今年股利政策將以現金股利為主搭配部分股票股利，配發率維持穩定。",
        "category": "dividend",
        "cluster_id": "CLU-2881-DIVIDEND-PROPOSAL",
        "revision_id": 1,
        "human_importance": 0.91,
    },

    # --------------------------------------------------------------------------
    # Multi-Outlet Syndicated Articles & Paraphrases (Within Cluster)
    # --------------------------------------------------------------------------
    {
        "event_id": "EV-2330-03-SYN",
        "article_id": "ART-2330-03-SYN",
        "stock_id": "2330",
        "source_name": "電子時報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-08-08",
        "pub_time_str": "18:45:00",
        "headline": "歐洲半導體製造核心成形台積電拍板在德勒斯登建12吋晶圓廠",
        "body_text": "台積電攜手歐系晶片巨頭成立合資晶圓廠，鎖定車用及工業微控制器市場，獲德國政府高額補貼支持。",
        "category": "capex",
        "cluster_id": "CLU-2330-GERMANY-FAB",  # Same cluster as EV-2330-03!
        "revision_id": 1,
        "human_importance": 0.91,
    },
    {
        "event_id": "EV-2603-06-SYN",
        "article_id": "ART-2603-06-SYN",
        "stock_id": "2603",
        "source_name": "經濟日報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2023-12-18",
        "pub_time_str": "17:40:00",
        "headline": "航運巨頭全面避開曼德海峽長榮宣布改道好望角確保航行安全",
        "body_text": "紅海局勢升溫威脅商船通行，長榮海運宣布跟進國際同業決策，旗下貨櫃船全數改繞非洲南端好望角以避開戰火風險。",
        "category": "crisis",
        "cluster_id": "CLU-2603-RED-SEA-REROUTE",  # Same cluster as EV-2603-06!
        "revision_id": 1,
        "human_importance": 0.96,
    },
    {
        "event_id": "EV-2881-06-SYN",
        "article_id": "ART-2881-06-SYN",
        "stock_id": "2881",
        "source_name": "自由時報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2024-01-10",
        "pub_time_str": "17:00:00",
        "headline": "富邦金全年獲利破660億每股盈餘4.81元創連15年獲利王紀錄",
        "body_text": "富邦金自結112年全年稅後淨利達661.37億元，EPS 4.81元續居金控龍頭寶座，北富銀與富邦人壽獲利雙引擎表現亮眼。",
        "category": "earnings",
        "cluster_id": "CLU-2881-FULLYEAR-2023",  # Same cluster as EV-2881-06!
        "revision_id": 1,
        "human_importance": 0.93,
    },

    # --------------------------------------------------------------------------
    # Revisions (Initial vs Clarification)
    # --------------------------------------------------------------------------
    {
        "event_id": "EV-2330-10-REV1",
        "article_id": "ART-2330-10-REV1",
        "stock_id": "2330",
        "source_name": "工商時報",
        "source_type": NewsSourceType.MEDIA_ARCHIVE,
        "date_str": "2024-03-18",
        "pub_time_str": "08:30:00",
        "headline": "傳台積電嘉義先進封裝廠投資規模擴大逾五千億元",
        "body_text": "市場傳聞台積電因應CoWoS產能需求緊繃，嘉義園區建廠計畫將擴大投資，預計興建多達六座封裝測試廠房。",
        "category": "capex",
        "cluster_id": "CLU-2330-CHIAYI-COWOS",
        "revision_id": 1,
        "lifecycle_status": "PUBLISHED",
        "human_importance": 0.88,
    },
    {
        "event_id": "EV-2330-10-REV2",
        "article_id": "ART-2330-10-REV2",
        "stock_id": "2330",
        "source_name": "MOPS",
        "source_type": NewsSourceType.MOPS_OFFICIAL,
        "date_str": "2024-03-18",
        "pub_time_str": "14:15:00",
        "headline": "說明媒體報導嘉義先進封裝建廠產能與投資規劃",
        "body_text": "台積電重申公司相關投資與產能建設計畫請以公開資訊觀測站公告為準，目前首期規劃兩座先進封裝廠，依市場需求審慎推動。",
        "category": "clarification",
        "cluster_id": "CLU-2330-CHIAYI-COWOS",
        "revision_id": 2,
        "lifecycle_status": "CLARIFIED",
        "human_importance": 0.70,
    },

    # --------------------------------------------------------------------------
    # Negative Controls & Routine Daily Market Commentary (Low Importance)
    # --------------------------------------------------------------------------
    {
        "event_id": "EV-2330-NEG-01",
        "article_id": "ART-2330-NEG-01",
        "stock_id": "2330",
        "source_name": "Yahoo奇摩股市",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2023-07-05",
        "pub_time_str": "13:45:00",
        "headline": "台積電收盤下跌2元收在582元成交量偏低呈整理走勢",
        "body_text": "台積電今日開高走低，終場小跌2元收在582元，成交量萎縮至兩萬餘張，市場觀望氣氛濃厚維持狹幅震盪整理。",
        "category": "routine",
        "cluster_id": "CLU-2330-ROUTINE-JUL23",
        "revision_id": 1,
        "human_importance": 0.15,
    },
    {
        "event_id": "EV-2603-NEG-01",
        "article_id": "ART-2603-NEG-01",
        "stock_id": "2603",
        "source_name": "Yahoo奇摩股市",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2023-09-20",
        "pub_time_str": "13:30:00",
        "headline": "長榮海運平盤作收航運類股交投清淡維持區間震盪",
        "body_text": "航運股今日缺乏消息面催化，長榮收在平盤112元，外資買賣超呈現互見，市場等待中國十一黃金週後裝船狀況。",
        "category": "routine",
        "cluster_id": "CLU-2603-ROUTINE-SEP23",
        "revision_id": 1,
        "human_importance": 0.12,
    },
    {
        "event_id": "EV-2881-NEG-01",
        "article_id": "ART-2881-NEG-01",
        "stock_id": "2881",
        "source_name": "Yahoo奇摩股市",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2023-10-25",
        "pub_time_str": "13:40:00",
        "headline": "富邦金成交量能平緩股價微跌0.2元守穩60元整數關卡",
        "body_text": "金控股今日呈現窄幅整理，富邦金股價波動僅數檔，終場小跌0.2元守穩60元大關，散戶投資人觀望美債殖利率走向。",
        "category": "routine",
        "cluster_id": "CLU-2881-ROUTINE-OCT23",
        "revision_id": 1,
        "human_importance": 0.14,
    },
    {
        "event_id": "EV-2330-NEG-02",
        "article_id": "ART-2330-NEG-02",
        "stock_id": "2330",
        "source_name": "Yahoo奇摩股市",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2023-11-20",
        "pub_time_str": "13:30:00",
        "headline": "外資小幅買超台積電股價力守580元平盤格局",
        "body_text": "外資今日小幅買超台積電約千餘張，晶圓代工龍頭維持區間震盪，市場聚焦輝達最新財報公布後的AI供應鏈反應。",
        "category": "routine",
        "cluster_id": "CLU-2330-ROUTINE-NOV23",
        "revision_id": 1,
        "human_importance": 0.18,
    },
    {
        "event_id": "EV-2603-NEG-02",
        "article_id": "ART-2603-NEG-02",
        "stock_id": "2603",
        "source_name": "Yahoo奇摩股市",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2024-01-22",
        "pub_time_str": "13:40:00",
        "headline": "長榮海運早盤走高尾盤漲幅收斂收在153元小漲0.5元",
        "body_text": "貨櫃三雄今日早盤受運價支撐走揚，惟獲利了結賣壓湧現，長榮終場僅微漲0.5元，成交張數較昨日明顯收斂。",
        "category": "routine",
        "cluster_id": "CLU-2603-ROUTINE-JAN24",
        "revision_id": 1,
        "human_importance": 0.20,
    },
    {
        "event_id": "EV-2881-NEG-02",
        "article_id": "ART-2881-NEG-02",
        "stock_id": "2881",
        "source_name": "Yahoo奇摩股市",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2024-02-20",
        "pub_time_str": "13:35:00",
        "headline": "富邦金新春開紅盤後連日小幅拉回終場收跌0.3元",
        "body_text": "金融保險指數走弱，富邦金開高走平終場收在66.2元小跌0.3元，金融股受長天期公債殖利率反彈影響略顯壓抑。",
        "category": "routine",
        "cluster_id": "CLU-2881-ROUTINE-FEB24",
        "revision_id": 1,
        "human_importance": 0.16,
    },
    {
        "event_id": "EV-2330-NEG-03",
        "article_id": "ART-2330-NEG-03",
        "stock_id": "2330",
        "source_name": "Yahoo奇摩股市",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2024-03-05",
        "pub_time_str": "13:30:00",
        "headline": "台積電盤中創高735元後逢高調節獲利了結收跌5元",
        "body_text": "台積電早盤衝高至735元歷史新高，午後獲利回吐賣壓出籠，終場反跌5元收在730元，高檔爆出逾五萬張大量。",
        "category": "routine",
        "cluster_id": "CLU-2330-ROUTINE-MAR24",
        "revision_id": 1,
        "human_importance": 0.22,
    },
    {
        "event_id": "EV-2603-NEG-03",
        "article_id": "ART-2603-NEG-03",
        "stock_id": "2603",
        "source_name": "Yahoo奇摩股市",
        "source_type": NewsSourceType.FINMIND_RSS,
        "date_str": "2024-03-25",
        "pub_time_str": "13:30:00",
        "headline": "長榮海運股息殖利率誘人但外資調節收跌1.5元",
        "body_text": "長榮海運每股配息10元獲散戶買盤支撐，惟外資在法說會後連續調節賣超，股價收在168元小跌1.5元維持整理。",
        "category": "routine",
        "cluster_id": "CLU-2603-ROUTINE-MAR24",
        "revision_id": 1,
        "human_importance": 0.25,
    },
]


# ==============================================================================
# 3. Pilot Dataset Builder
# ==============================================================================

class PilotDatasetBuilder:
    """
    Builds the complete V8.10 Supervised Pilot Dataset:
    1. Loads verified historical market DataFrames (2330, 2603, 2881, TAIEX)
    2. Enforces TWSE Trading Calendar reconciliation
    3. Computes pre-T 32D event metadata vector and genuine 128D semantic text vector
    4. Computes authentic ex-post market reaction target labels from T+1/T+2 market data
    5. Formulates normalized composite target importance score y in [0, 1]
    6. Verifies strict holdout isolation (4 layers: exact, headline, near-dup, cluster)
    """

    def __init__(
        self,
        data_dir: str = "data/poc_real_data",
        encoder: Optional[GenuinePretrainedSemanticEncoder] = None,
    ):
        self.data_dir = data_dir
        self.market_dfs: Dict[str, pd.DataFrame] = {}
        self.taiex_df: Optional[pd.DataFrame] = None
        self._load_market_data()
        
        # Load or use existing semantic encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = load_trusted_pretrained_embedding_engine(local_files_only=True)

    def _load_market_data(self) -> None:
        """Loads and indexes authentic OHLCV DataFrames."""
        for stock_id in ["2330", "2603", "2881"]:
            fpath = os.path.join(self.data_dir, f"{stock_id}_ohlcv.csv")
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                df["dt"] = pd.to_datetime(df["date"])
                df = df.set_index("dt").sort_index()
                self.market_dfs[stock_id] = df
            else:
                raise FileNotFoundError(f"Missing market OHLCV file: {fpath}")

        taiex_path = os.path.join(self.data_dir, "taiex.csv")
        if os.path.exists(taiex_path):
            tdf = pd.read_csv(taiex_path)
            tdf["dt"] = pd.to_datetime(tdf["date"])
            tdf = tdf.set_index("dt").sort_index()
            self.taiex_df = tdf
        else:
            raise FileNotFoundError(f"Missing TAIEX index file: {taiex_path}")

        # Reconcile calendar
        reconciled = TWSETradingCalendar.reconcile_with_market_index(self.taiex_df.index)
        if not reconciled["is_reconciled"]:
            raise ValueError(f"TWSE Trading Calendar failed reconciliation: {reconciled}")

    def build_dataset(
        self,
        raw_events: Optional[List[Dict[str, Any]]] = None,
        verify_holdout: bool = True,
    ) -> List[PilotEventSample]:
        """
        Builds the supervised pilot dataset across all raw curated events.
        """
        events = raw_events or RAW_PILOT_EVENTS
        samples: List[PilotEventSample] = []

        # 1. First verify holdout contamination
        if verify_holdout:
            audit_result = verify_holdout_contamination(events, SEMANTIC_EVALUATION_BENCHMARK)
            if audit_result["is_contaminated"]:
                raise RuntimeError(
                    f"V8.10 Holdout Contamination Detected in Pilot Candidates! "
                    f"Details: {audit_result['details']}"
                )

        # 2. Process each event
        for ev in events:
            stock_id = ev["stock_id"]
            d_str = ev["date_str"]
            pub_time_str = ev["pub_time_str"]

            t_decision = pd.Timestamp(f"{d_str} 20:00:00", tz="Asia/Taipei")
            t_published = pd.Timestamp(f"{d_str} {pub_time_str}", tz="Asia/Taipei")
            t_available = t_published  # Estimated-PIT: published time == available time

            stock_df = self.market_dfs[stock_id]
            market_df = self.taiex_df

            # Pre-T 32D Metadata Feature Extraction
            art = RawNewsArticle(
                source_type=ev["source_type"],
                source_name=ev["source_name"],
                stock_id=stock_id,
                headline=ev["headline"],
                body_text=ev["body_text"],
                source_published_at=t_published,
                provider_available_at=t_available,
                revision_id=ev.get("revision_id", 1),
                lifecycle_status=ev.get("lifecycle_status", "PUBLISHED"),
                evidence_audit_level=EvidenceAuditLevel.ESTIMATED,
                article_id=ev["article_id"],
            )
            meta_32d = extract_event_metadata_vector(art, stock_id, t_decision).numpy()

            # Pretrained 128D Semantic Text Embedding
            full_text = f"{ev['headline']} {ev['body_text']}"
            text_128d = self.encoder.embed_text(full_text)

            # Ex-Post Market Reaction Target Label (T+1/T+2)
            # Strictly authentic: Beta120 and Volume stats computed strictly <= T!
            label = ExPostMarketReactionLabel.compute_from_market_data(
                stock_id=stock_id,
                t_decision=t_decision,
                stock_df=stock_df,
                market_df=market_df,
                calendar=TWSETradingCalendar,
            )

            # Formulate continuous target score y in [0, 1]
            # Balanced synthesis of prior human domain importance (0.4) and ex-post market reaction (0.6)
            # If market reaction is missing (e.g. edge of dataset), fall back to human importance
            human_imp = float(ev.get("human_importance", 0.5))
            if label.missing_status == "AVAILABLE" and label.composite_market_impact_score is not None:
                market_score = label.composite_market_impact_score
                target_score = round(float(np.clip(0.4 * human_imp + 0.6 * market_score, 0.0, 1.0)), 4)
            else:
                target_score = round(human_imp, 4)

            sample = PilotEventSample(
                sample_id=f"SMP-{ev['event_id']}",
                event_id=ev["event_id"],
                article_id=ev["article_id"],
                stock_id=stock_id,
                source_name=ev["source_name"],
                source_type=ev["source_type"],
                decision_timestamp=t_decision,
                published_at=t_published,
                available_at=t_available,
                headline=ev["headline"],
                body_text=ev["body_text"],
                category=ev["category"],
                cluster_id=ev["cluster_id"],
                revision_id=ev.get("revision_id", 1),
                lifecycle_status=ev.get("lifecycle_status", "PUBLISHED"),
                evidence_level=EvidenceAuditLevel.ESTIMATED,
                provenance_category="ESTIMATED_PIT_NEWS",
                human_importance=human_imp,
                ex_post_label=label,
                target_score=target_score,
                metadata_32d=meta_32d,
                text_128d=text_128d,
            )
            sample.verify_causality()
            samples.append(sample)

        # Sort chronologically by decision_timestamp
        samples.sort(key=lambda s: s.decision_timestamp)
        return samples
