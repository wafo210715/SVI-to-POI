# WeMap POI Dataset & SVI-POI Extraction Research Summary

**Date:** 2026-01-24
**Project:** HCMC Spatial-Semantic POI Reconstruction

---

## 1. WeMap POI Dataset

### Overview
- **Platform:** WeMap (wemap.vn) - Vietnam's #1 Digital Map
- **Developer:** FIMO JSC
- **Government Project:** Part of "Digital Vietnam Knowledge System" (Hệ tri thức Việt số hóa)
- **Research Paper:** [Large-scale Vietnamese point-of-interest classification using weak labeling](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.1020532/full) (Frontiers in AI, 2022)

### Dataset Availability
- **GitHub Repository:** [PIVASIA/wemap-poi-dataset](https://github.com/PIVASIA/wemap-poi-dataset)
- **Files:**
  - `raw/test.json` (2.2 MB) - 17,651 gold-standard POIs
  - `raw/train.json` (75 MB) - Raw crowd-sourced data
  - `weak_labeled/train.json` (30 MB) - 275K weak-labeled POIs
- **License:** Research/educational use only (copyright FIMO JSC 2022)

### Data Structure
```json
{
  "_id": "5ceb48cacd72f703274ffab4",
  "name": "Ao",                    // Vietnamese POI name
  "poicat": "Ao, hồ",              // Original category (158 types)
  "validate": 1                    // Validation flag
}
```

**Limitations:**
- ❌ NO temporal/yearly information
- ❌ NO GPS coordinates (lat/lon)
- ❌ NO address fields
- ❌ NO image references

### 15-Category Schema (Pelias-based)

| Category | Vietnamese Examples |
|----------|---------------------|
| **food** | Quán ăn bình dân, Nhà hàng, Cửa hàng tạp hóa |
| **retail** | Cửa hàng thời trang, Siêu thị, Chuỗi cửa hàng tiện ích |
| **health** | Bệnh viện đa khoa, Hiệu thuốc, Phòng khám |
| **education** | Trường tiểu học, Đại học, Trung tâm ngoại ngữ |
| **government** | UBND, Sở Bộ Ngành, Công an quốc phòng |
| **finance** | Ngân hàng, Cây ATM, Công ty chứng khoán |
| **transport** | Bến xe, Trạm xăng, Ga tàu, Bãi đỗ xe |
| **accommodation** | Khách sạn, Home stay, Resort |
| **religion** | Chùa, nhà thờ, đình đền |
| **entertainment** | Rạp chiếu phim, Quán bar, Trung tâm vui chơi |
| **recreation** | Sân vận động, Gym, Công viên |
| **natural** | Ao hồ, Điểm du lịch tự nhiên |
| **industry** | Nhà máy, xí nghiệp |
| **professional** | Văn phòng luật, dịch vụ |
| **nightlife** | Bar, karaoke, vũ trường |

---

## 2. Digital Vietnam Knowledge System

### Overview
- **Official Name:** Hệ tri thức Việt số hóa (Digital Vietnamese Knowledge System)
- **Launch Date:** January 1, 2018
- **Government Approval:** Decision 677/QĐ-TTg (May 18, 2017)
- **Portal:** [itrithuc.vn](https://itrithuc.vn/)
- **Lead Ministry:** Ministry of Science and Technology (MOST)

### Four Main Components
1. **Open Database** - Public data from ministries, agencies, localities, businesses
2. **Q&A Banks** - Question and answer system
3. **Enterprise Archives** - Business information database
4. **Developers** - API and application ecosystem

### Key Statistics (as of 2019)
- **23.4 million addresses** collected (95% of Vietnam)
- **63/63 provinces** completed
- **Partners:** VNPost, Vietnam National University

### Free Services
- Website: https://itrithuc.vn/
- Phone: Dial 1001 (Viettel, Mobiphone, Vinaphone)

---

## 3. Mapillary Street View Temporal Analysis

### Temporal Metadata
- **Field:** `captured_at` (timestamp in milliseconds since epoch)
- **Source:** Extracted from image EXIF data during upload
- **Availability:** Available via API v4 entity endpoints

### Update Frequency
**Mapillary is crowdsourced** - NO fixed update schedule:
- ❌ No guaranteed yearly updates
- ❌ Highly location-dependent (urban areas > rural)
- ❌ Variable temporal resolution
- ❌ ~50% of historical images may lack timestamp info

### Implications for HCMC POI Pipeline
| Factor | Impact |
|--------|--------|
| **Crowdsourced nature** | Can't predict when new images will appear |
| **Temporal variance** | Some locations may have frequent updates, others stagnant for years |
| **Missing timestamps** | Some historical data unusable for temporal analysis |
| **Research value** | Still valuable - can track changes when new images DO appear |

**Recommendation:** Build pipeline to query `captured_at` timestamps and analyze actual revisit rates for specific HCMC areas of interest.

---

## 4. Existing SVI-to-POI Research & Implementations

### Production Systems (Deployed)

#### DuMapper (Baidu Maps) ⭐
- **Paper:** [DuMapper: Towards Automatic Verification of Large-Scale POIs with Street Views](https://arxiv.org/html/2411.18073v1) (CIKM 2022)
- **Status:** ✅ **In production since June 2018**
- **Scale:** 405 million POI verifications (2018-2021) = ~800 expert mappers
- **Method:**
  - **DuMapper I:** Geo-spatial index + OCR + Candidate ranking
  - **DuMapper II:** Deep multimodal embedding + ANN search (50x faster)
- **Accuracy:** SR@1 = 91.74% (automatic) vs 94.52% (human expert)
- **Input:** Signboard image + coordinates
- **Output:** POI verification from database

### Academic Frameworks

#### SVI2POI (2026)
- **Source:** [Taylor & Francis Journal](https://www.tandfonline.com/doi/full/10.1080/20964471.2025.2600170)
- **Focus:** End-to-end framework with signboard recognition stage
- **Novelty:** Structural features for urban characteristics

#### Semantic-Rich Location Search (2024)
- **Source:** [ACM DL](https://dl.acm.org/doi/10.1145/3681769.3698583)
- **Focus:** POI-street view matching accuracy

#### Multimodal POI Semantic Annotation (IJCAI 2024)
- **Source:** [IJCAI Proceedings](https://www.ijcai.org/proceedings/2024/0280.pdf)
- **Focus:** Semantic features + spatial neighbor information

#### OpenFACADES (2025)
- **Source:** [arXiv](https://arxiv.org/html/2504.02866v1)
- **Focus:** Architectural caption and attribute extraction

### Data Collection Tools
- **GitHub:** [CollectGISData](https://github.com/kkyyhh96/CollectGISData)
  - Baidu API for POI data
  - Tencent for street view images

### Foundational Research
- **Urban Visual Intelligence** (PNAS 2023) - 174 citations
- **YOLO-based POI extraction** (ISPRS 2022)

---

## 5. Where SVI-POI Has Been Applied

### Geographic Coverage
| Region | System | Status |
|--------|--------|--------|
| **China** | Baidu DuMapper | ✅ Production (2018-present) |
| **Global** | Google Street View | ✅ Limited (post-2017 depth only) |
| **Global** | Mapillary | ✅ Crowdsourced, variable coverage |
| **Vietnam** | WeMap | ✅ POI database (no direct SVI extraction) |

### Application Domains
1. **Map Maintenance & Verification** (DuMapper)
2. **Urban Analytics** (Urban Visual Intelligence)
3. **Change Detection** (bi-temporal analysis)
4. **Accessibility Assessment** (GEDIT at Baidu Maps)
5. **Traffic Prediction** (DuTraffic)

---

## 6. Key Takeaways for HCMC Project

### ✅ Confirmed
- WeMap dataset exists with Vietnam-specific 15-category schema
- DuMapper proves SVI-POI extraction is production-viable
- Mapillary has `captured_at` timestamps (when available)

### ❌ Limitations
- WeMap dataset has NO coordinates or temporal data
- Mapillary update frequency is NOT guaranteed
- No existing Vietnam-specific SVI-POI implementation found

### 🎯 Strategic Advantages
- **First Vietnam-specific SVI-POI pipeline** (research gap identified)
- Can combine WeMap's schema with Mapillary's temporal metadata
- VLM-based approach (GLM-4V) is novel vs traditional OCR (DuMapper)

---

## 7. Updated POI Schema Recommendation

**Use WeMap's 15-category schema** (not CLAUDE.md's business-only schema):

```json
{
  "poi_name_vietnamese": "string",
  "poi_name_english": "string",
  "category": "food|retail|health|education|government|finance|transport|accommodation|religion|entertainment|recreation|natural|industry|professional|nightlife",
  "sub_category": "string",  // Optional: from 158 Vietnamese types
  "confidence_score": 0-1,
  "has_signboard": boolean,
  "signboard_text_detected": "string",
  "no_poi_detected": boolean  // For images with no identifiable POI
}
```

---

## References

### Datasets
- [WeMap POI Dataset - GitHub](https://github.com/PIVASIA/wemap-poi-dataset)
- [Large-scale Vietnamese POI classification - Frontiers in AI](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.1020532/full)

### Government
- [Digital Vietnamese Knowledge System Launch - MIC](https://english.mic.gov.vn/digital-vietnamese-knowledge-system-launched-197136370.htm)
- [itrithuc.vn](https://itrithuc.vn/)

### Research Papers
- [DuMapper - arXiv](https://arxiv.org/html/2411.18073v1)
- [SVI2POI - Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/20964471.2025.2600170)
- [Semantic-Rich Location Search - ACM](https://dl.acm.org/doi/10.1145/3681769.3698583)
- [Multimodal POI Annotation - IJCAI 2024](https://www.ijcai.org/proceedings/2024/0280.pdf)
- [Urban Visual Intelligence - PNAS](https://www.pnas.org/doi/10.1073/pnas.2220417120)

### Tools
- [CollectGISData - GitHub](https://github.com/kkyyhh96/CollectGISData)
- [Mapillary API Documentation](https://www.mapillary.com/developer/api-documentation)

### Platforms
- [WeMap Vietnam](https://wemap.vn/)
- [FIMO JSC](https://fimo.vn/)
