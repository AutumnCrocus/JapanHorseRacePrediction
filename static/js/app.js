/**
 * 競馬予想AI - フロントエンドアプリケーション
 */

// API エンドポイント
const API_BASE = '';

// DOM要素
const elements = {
    demoBtn: document.getElementById('demoBtn'),
    predictionResults: document.getElementById('predictionResults'),
    loading: document.getElementById('loading'),
    raceName: document.getElementById('raceName'),
    raceData01: document.getElementById('raceData01'),
    raceData02: document.getElementById('raceData02'),
    raceDetails: document.getElementById('raceDetails'),
    timestamp: document.getElementById('timestamp'),
    topHorses: document.getElementById('topHorses'),
    predictionsTable: document.getElementById('predictionsTable'),
    featureImportance: document.getElementById('featureImportance'),
    raceUrl: document.getElementById('raceUrl'),
    predictUrlBtn: document.getElementById('predictUrlBtn'),
    budget: document.getElementById('budget'),
    // モデル情報要素
    modelAlgo: document.getElementById('modelAlgo'),
    modelTarget: document.getElementById('modelTarget'),
    modelSource: document.getElementById('modelSource'),
    modelFeatures: document.getElementById('modelFeatures'),
    // IPAT連携要素
    ipatConnectBtn: document.getElementById('ipatConnectBtn'),
    ipatLoginModal: document.getElementById('ipatLoginModal'),
    ipatLoginForm: document.getElementById('ipatLoginForm'),
    ipatVoteConfirmModal: document.getElementById('ipatVoteConfirmModal'),
    confirmVoteBtn: document.getElementById('confirmVoteBtn')
};

/**
 * 初期化
 */
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadFeatureImportance();
    loadModelInfo();
});

/**
 * イベントリスナーを設定
 */
function initEventListeners() {
    elements.demoBtn.addEventListener('click', runDemo);
    if (elements.predictUrlBtn) {
        elements.predictUrlBtn.addEventListener('click', runUrlPrediction);
    }

    // IPAT連携ボタン
    if (elements.ipatConnectBtn) {
        elements.ipatConnectBtn.addEventListener('click', handleIpatConnect);
    }

    // IPATログインフォーム
    if (elements.ipatLoginForm) {
        elements.ipatLoginForm.addEventListener('submit', handleIpatLogin);
    }

    // IPAT投票確認ボタン
    if (elements.confirmVoteBtn) {
        elements.confirmVoteBtn.addEventListener('click', handleConfirmVote);
    }

    // ナビゲーションのスムーススクロール
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = e.target.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: 'smooth' });
            }
            // アクティブ状態を更新
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            e.target.classList.add('active');
        });
    });
}

/**
 * デモを実行
 */
async function runDemo() {
    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/demo`);
        const data = await response.json();

        if (data.success) {
            displayResults(data);
            scrollToPredictions();
        } else {
            showError(data.error || '予測に失敗しました');
        }
    } catch (error) {
        console.error('Demo error:', error);
        // APIが利用できない場合はモックデータを使用
        displayResults({
            success: true,
            predictions: [
                { predicted_rank: 1, horse_number: 6, horse_name: 'イクイノックス', probability: 0.72, odds: 1.8, popularity: 1, expected_value: 1.30 },
                { predicted_rank: 2, horse_number: 1, horse_name: 'ディープインパクト', probability: 0.65, odds: 2.5, popularity: 2, expected_value: 1.63 },
                { predicted_rank: 3, horse_number: 4, horse_name: 'アーモンドアイ', probability: 0.58, odds: 3.2, popularity: 3, expected_value: 1.86 }
            ],
            race_name: 'デモレース - 日本ダービー（G1）',
            race_data01: '15:40発走 / 芝2400m (左) / 天候:晴 / 馬場:良',
            race_data02: '2回 東京 12日目 サラ系３歳 オープン',
            timestamp: new Date().toISOString()
        });
        scrollToPredictions();
    } finally {
        showLoading(false);
    }
}

/**
 * 予測を実行
 */
async function runPrediction() {
    const inputData = elements.horseData.value.trim();

    if (!inputData) {
        showError('馬データを入力してください');
        return;
    }

    let horses;
    try {
        horses = JSON.parse(inputData);
    } catch (e) {
        showError('JSONの形式が正しくありません');
        return;
    }

    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ horses })
        });

        const data = await response.json();

        if (data.success) {
            displayResults({
                ...data,
                race_name: 'カスタムレース',
                race_data01: '入力データによる予測',
                race_data02: ''
            });
            scrollToPredictions();
        } else {
            showError(data.error || '予測に失敗しました');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError('予測処理中にエラーが発生しました');
    } finally {
        showLoading(false);
    }
}

/**
 * URLから予測を実行
 */
async function runUrlPrediction() {
    const url = elements.raceUrl.value.trim();

    if (!url) {
        showError('URLを入力してください');
        return;
    }

    if (!url.includes('race_id=')) {
        showError('有効なNetkeibaのレースURLを入力してください');
        return;
    }

    showLoading(true);

    const budget = document.getElementById('budget') ? document.getElementById('budget').value : 0;

    try {
        const response = await fetch(`${API_BASE}/api/predict_by_url`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url, budget })
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            scrollToPredictions();
        } else {
            showError(data.error || '予測に失敗しました');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError('予測処理中にエラーが発生しました: ' + error.message);
    } finally {
        showLoading(false);
    }
}

/**
 * 結果を表示
 */
// 最新の予測結果データを保持
let currentRaceId = null;

/**
 * 結果を表示
 */
function displayResults(data) {
    const { predictions, race_name, race_info, race_data01, race_data02, timestamp, odds_warning, race_id } = data;

    // データをグローバル変数に保持
    currentRaceId = race_id;
    currentRecommendations = data.recommendations || [];

    // ヘッダー情報
    elements.raceName.textContent = race_name || 'レース予測結果';
    if (elements.raceData01) elements.raceData01.textContent = race_data01 || '';
    if (elements.raceData02) elements.raceData02.textContent = race_data02 || '';
    elements.raceDetails.textContent = race_info || '';
    elements.timestamp.textContent = formatTimestamp(timestamp);

    // トップ3を表示
    displayTopHorses(predictions.slice(0, 3));

    // 全馬テーブルを表示
    displayPredictionsTable(predictions);

    // オッズ警告を表示
    const oddsWarningSection = document.getElementById('oddsWarningSection');
    if (odds_warning) {
        const oddsWarningMessage = document.getElementById('oddsWarningMessage');
        if (oddsWarningMessage) {
            oddsWarningMessage.textContent = odds_warning;
        }
        if (oddsWarningSection) {
            oddsWarningSection.classList.remove('hidden');
        }
    } else {
        if (oddsWarningSection) {
            oddsWarningSection.classList.add('hidden');
        }
    }

    // 推奨買い目を表示
    if (data.recommendations && data.recommendations.length > 0) {
        displayRecommendations(data.recommendations);
        document.getElementById('recommendationSection').classList.remove('hidden');

        // 自信度バッジの表示
        const confidenceEl = document.getElementById('confidenceLevel');
        if (confidenceEl && data.confidence_level) {
            const level = data.confidence_level;
            confidenceEl.textContent = `自信度: ${level}`;
            confidenceEl.classList.remove('hidden');

            // クラスのリセット
            confidenceEl.className = 'confidence-badge';

            // 色分け
            if (level === 'S') confidenceEl.classList.add('confidence-s');
            else if (level === 'A') confidenceEl.classList.add('confidence-a');
            else if (level === 'B') confidenceEl.classList.add('confidence-b');
            else if (level === 'C') confidenceEl.classList.add('confidence-c');
            else confidenceEl.classList.add('confidence-d');
        }

    } else {
        document.getElementById('recommendationSection').classList.add('hidden');
    }

    // 結果セクションを表示
    elements.predictionResults.classList.remove('hidden');
}

/**
 * トップ3馬を表示
 */
function displayTopHorses(topHorses) {
    elements.topHorses.innerHTML = topHorses.map((horse, index) => `
        <div class="top-horse-card rank-${index + 1}">
            <div class="rank-badge">${index + 1}</div>
            <div class="horse-number">${horse.horse_number}番</div>
            <div class="horse-name">${horse.horse_name}</div>
            <div class="horse-stats">
                <div class="stat-item">
                    <span class="stat-label">複勝確率</span>
                    <span class="stat-value highlight">${formatPercent(horse.probability)}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">単勝オッズ</span>
                    <span class="stat-value">${horse.odds.toFixed(1)}倍</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">人気</span>
                    <span class="stat-value">${horse.popularity}番人気</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">期待値</span>
                    <span class="stat-value ${horse.expected_value >= 1 ? 'highlight' : ''}">${horse.expected_value.toFixed(2)}</span>
                </div>
            </div>
        </div>
    `).join('');
}

/**
 * 予測テーブルを表示
 */
function displayPredictionsTable(predictions) {
    elements.predictionsTable.innerHTML = predictions.map(horse => {
        // バッジの生成
        let badgeHtml = '';
        if (horse.analysis && horse.analysis.type !== 'normal') {
            const badgeClass = horse.analysis.type === 'danger' ? 'badge-danger' : 'badge-value';
            const icon = horse.analysis.type === 'danger' ? '⚠️' : '⭐';
            badgeHtml = `<span class="analysis-badge ${badgeClass}" title="${horse.analysis.message}">${icon} ${horse.analysis.message}</span>`;
        }

        return `
        <tr>
            <td><strong>${horse.predicted_rank}位</strong></td>
            <td>${horse.horse_number}</td>
            <td>
                ${horse.horse_name}
                ${badgeHtml}
                ${horse.reasoning && (horse.reasoning.positive.length > 0 || horse.reasoning.negative.length > 0) ?
                `<button class="reasoning-btn" onclick="showReasoning(${horse.horse_number}, '${horse.horse_name.replace(/'/g, "\\'")}', ${JSON.stringify(horse.reasoning).replace(/"/g, '&quot;')})">
                        💡
                    </button>` : ''
            }
            </td>
            <td>
                ${formatPercent(horse.probability)}
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${horse.probability * 100}%"></div>
                </div>
            </td>
            <td>${horse.odds.toFixed(1)}倍</td>
            <td class="${horse.expected_value >= 1 ? 'highlight' : ''}">${horse.expected_value.toFixed(2)}</td>
        </tr>
    `}).join('');
}

// 特徴量名を日本語にマッピング
const FEATURE_LABELS = {
    '単勝': '単勝オッズ',
    '人気': '人気順位',
    'avg_rank': '平均着順',
    'win_rate': '勝率',
    'place_rate': '複勝率',
    'jockey_avg_rank': '騎手平均着順',
    'jockey_win_rate': '騎手勝率',
    'avg_last_3f': '平均上がり3F',
    'avg_running_style': '脚質（位置取り）',
    '枠番': '枠番',
    '馬番': '馬番',
    '斤量': '斤量',
    '年齢': '年齢',
    '体重': '馬体重',
    '体重変化': '馬体重変化',
    'course_len': 'コース距離',
    'race_count': '出走回数',
    'venue_id': '競馬場',
    '性': '性別',
    'race_type': 'コース種別',
    'kai': '開催回',
    'day': '開催日',
    'race_num': 'レース番号'
};

// 予想根拠を表示するモーダル
function showReasoning(horseNumber, horseName, reasoning) {
    const positiveFactors = reasoning.positive || [];
    const negativeFactors = reasoning.negative || [];

    const positiveHtml = positiveFactors.length > 0 ? `
        <div class="factors-section positive">
            <h4>✅ プラス要因</h4>
            <ul>
                ${positiveFactors.map(f => `
                    <li>
                        <strong>${FEATURE_LABELS[f.feature] || f.feature}</strong>: 
                        ${f.value.toFixed(2)}
                        <span class="contribution positive">+${Math.abs(f.contribution).toFixed(3)}</span>
                    </li>
                `).join('')}
            </ul>
        </div>
    ` : '<p>プラス要因が検出されませんでした。</p>';

    const negativeHtml = negativeFactors.length > 0 ? `
        <div class="factors-section negative">
            <h4>⚠️ マイナス要因</h4>
            <ul>
                ${negativeFactors.map(f => `
                    <li>
                        <strong>${FEATURE_LABELS[f.feature] || f.feature}</strong>: 
                        ${f.value.toFixed(2)}
                        <span class="contribution negative">${f.contribution.toFixed(3)}</span>
                    </li>
                `).join('')}
            </ul>
        </div>
    ` : '<p>マイナス要因が検出されませんでした。</p>';

    const modalHtml = `
        <div class="modal-overlay" onclick="closeReasoning()">
            <div class="modal-content" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h3>${horseName}（${horseNumber}番）の予想根拠</h3>
                    <button class="close-btn" onclick="closeReasoning()">×</button>
                </div>
                <div class="modal-body">
                    <p class="modal-description">
                        AIがこの馬の複勝確率を算出する際に、特に影響が大きかった要素を表示しています。
                    </p>
                    ${positiveHtml}
                    ${negativeHtml}
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

// 根拠モーダルを閉じる
function closeReasoning() {
    const modal = document.querySelector('.modal-overlay');
    if (modal) {
        modal.remove();
    }
}

/**
 * 特徴量重要度を読み込み
 */
async function loadFeatureImportance() {
    try {
        const response = await fetch(`${API_BASE}/api/feature_importance`);
        const data = await response.json();

        if (data.success) {
            if (data.available) {
                displayFeatureImportance(data.features);
            } else {
                elements.featureImportance.innerHTML = `<div class="placeholder-message"><p>ℹ️ ${data.message || '特徴量重要度は利用できません'}</p></div>`;
            }
        } else {
            // API returned success: false
            const errorMessage = data.message || data.error || 'データの読み込みに失敗しました';
            elements.featureImportance.innerHTML = `<div class="placeholder-message error"><p>⚠️ ${errorMessage}</p></div>`;
        }
    } catch (error) {
        console.error('Feature importance error:', error);
        elements.featureImportance.innerHTML = `<div class="placeholder-message error"><p>⚠️ 通信エラーが発生しました</p></div>`;
    }
}

/**
 * モデル情報を読み込み
 */
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/api/model_info`);
        const data = await response.json();

        if (data.success) {
            elements.modelAlgo.textContent = data.algorithm;
            elements.modelTarget.textContent = data.target;
            elements.modelSource.textContent = data.source;
            elements.modelFeatures.textContent = `${data.feature_count}種類`;
        } else {
            const errorText = '読み込み失敗';
            elements.modelAlgo.textContent = errorText;
            elements.modelTarget.textContent = errorText;
            elements.modelSource.textContent = errorText;
            elements.modelFeatures.textContent = '-';
        }
    } catch (error) {
        console.error('Model info error:', error);
        const errorText = '通信エラー';
        elements.modelAlgo.textContent = errorText;
        elements.modelTarget.textContent = errorText;
        elements.modelSource.textContent = errorText;
    }
}

/**
 * 特徴量重要度を表示
 */
function displayFeatureImportance(features) {
    if (!features || features.length === 0) {
        features = getMockFeatureImportance();
    }

    const maxImportance = Math.max(...features.map(f => f.importance));

    elements.featureImportance.innerHTML = features.slice(0, 10).map(feature => `
        <div class="feature-bar">
            <span class="feature-name">${translateFeatureName(feature.feature)}</span>
            <div class="feature-bar-container">
                <div class="feature-bar-fill" style="width: ${(feature.importance / maxImportance) * 100}%"></div>
            </div>
            <span class="feature-value">${feature.importance.toFixed(0)}</span>
        </div>
    `).join('');
}

/**
 * 特徴量名を日本語に変換
 */
function translateFeatureName(name) {
    const translations = {
        '人気': '人気順',
        '単勝': '単勝オッズ',
        'avg_rank': '平均着順',
        'win_rate': '勝率',
        'place_rate': '複勝率',
        'race_count': '出走回数',
        'jockey_avg_rank': '騎手平均着順',
        'jockey_win_rate': '騎手勝率',
        '斤量': '斤量',
        '年齢': '年齢',
        '体重': '馬体重',
        '体重変化': '体重増減',
        'course_len': 'コース距離',
        '枠番': '枠番',
        '馬番': '馬番',
        '性': '性別',
        'race_type': 'コース種別'
    };
    return translations[name] || name;
}

/**
 * モック結果を表示（APIが利用できない場合）
 */
function displayMockResults() {
    const mockData = {
        success: true,
        predictions: [
            { predicted_rank: 1, horse_number: 6, horse_name: 'イクイノックス', probability: 0.72, odds: 1.8, popularity: 1, expected_value: 1.30 },
            { predicted_rank: 2, horse_number: 1, horse_name: 'ディープインパクト', probability: 0.65, odds: 2.5, popularity: 2, expected_value: 1.63 },
            { predicted_rank: 3, horse_number: 4, horse_name: 'アーモンドアイ', probability: 0.58, odds: 3.2, popularity: 3, expected_value: 1.86 },
            { predicted_rank: 4, horse_number: 5, horse_name: 'コントレイル', probability: 0.52, odds: 4.5, popularity: 4, expected_value: 2.34 },
            { predicted_rank: 5, horse_number: 2, horse_name: 'オルフェーヴル', probability: 0.48, odds: 5.8, popularity: 4, expected_value: 2.78 },
            { predicted_rank: 6, horse_number: 3, horse_name: 'キタサンブラック', probability: 0.42, odds: 8.2, popularity: 5, expected_value: 3.44 },
            { predicted_rank: 7, horse_number: 7, horse_name: 'リバティアイランド', probability: 0.38, odds: 12.0, popularity: 6, expected_value: 4.56 },
            { predicted_rank: 8, horse_number: 8, horse_name: 'ドゥラメンテ', probability: 0.30, odds: 15.0, popularity: 7, expected_value: 4.50 }
        ],
        race_name: 'デモレース - 日本ダービー（G1）',
        race_info: '芝2400m / 良',
        timestamp: new Date().toISOString()
    };

    displayResults(mockData);
}

/**
 * モック特徴量重要度
 */
function getMockFeatureImportance() {
    return [
        { feature: '人気', importance: 2500 },
        { feature: 'avg_rank', importance: 2200 },
        { feature: '単勝', importance: 1800 },
        { feature: 'win_rate', importance: 1600 },
        { feature: 'place_rate', importance: 1400 },
        { feature: 'jockey_win_rate', importance: 1200 },
        { feature: '斤量', importance: 1000 },
        { feature: 'race_count', importance: 900 },
        { feature: '年齢', importance: 800 },
        { feature: 'course_len', importance: 700 }
    ];
}

/**
 * ローディング表示を切り替え
 */
function showLoading(show) {
    if (show) {
        elements.loading.classList.remove('hidden');
        elements.predictionResults.classList.add('hidden');
    } else {
        elements.loading.classList.add('hidden');
    }
}

/**
 * エラーを表示
 */
function showError(message) {
    alert(message);
}

/**
 * 予測結果へスクロール
 */
function scrollToPredictions() {
    setTimeout(() => {
        elements.predictionResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * パーセント表示にフォーマット
 */
function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
}

/**
 * タイムスタンプをフォーマット
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleString('ja-JP');
}

/**
 * 推奨買い目を表示
 */
/**
 * 推奨買い目を表示
 */
function displayRecommendations(recommendations) {
    const tbody = document.getElementById('recommendationTableBody');
    if (!tbody) return;

    tbody.innerHTML = recommendations.map(rec => {
        // BettingAllocator format
        const type = rec.bet_type || rec.type;
        const combo = rec.combination || rec.combo || rec.umaban;
        const desc = rec.description || rec.desc || (rec.method === 'BOX' ? 'BOX' : '');
        const amount = rec.total_amount || rec.amount;
        const pts = rec.points || 1;
        const reason = rec.reason || '-';

        // 旧フォーマット互換 (evなど)
        const ev = rec.ev !== undefined ? rec.ev.toFixed(2) : '-';
        const prob = rec.prob !== undefined ? formatPercent(rec.prob) : '-';
        const odds = rec.odds !== undefined ? rec.odds.toFixed(1) + '倍' : '-';

        const money = amount ? `¥${amount.toLocaleString()}` : '-';

        return `
        <tr>
            <td>
                <span class="badge badge-primary">${type}</span>
                <span style="font-size:0.8em; margin-left:4px; color:#666;">${desc}</span>
            </td>
            <td><strong>${combo}</strong> <small style="color:#888;">(${pts}点)</small></td>
            <td>-</td> <!-- 馬名はBOX等の場合複数になるため省略 -->
            <td>${odds}</td>
            <td>${prob}</td>
            <td>${ev}</td>
            <td class="money">${money}</td>
            <td class="reason"><small>${reason}</small></td>
        </tr>
        `;
    }).join('');
}

// ========================================
// IPAT連携関連の関数 (Selenium Browser Automation)
// ========================================

// グローバル変数
let currentRecommendations = [];

/**
 * 券種名とIPATコードの対応マップ
 */
const BET_TYPE_CODES = {
    '単勝': 1,
    '複勝': 2,
    // その他は未対応（netkeibaオートメーション側で未実装のため）
    '枠連': 3, '馬連': 4, 'ワイド': 5, '馬単': 6, '3連複': 7, '3連単': 8
};

/**
 * IPAT連携ボタンクリック時の処理
 */
function handleIpatConnect() {
    // 推奨買い目が表示されているか確認
    if (!currentRecommendations || currentRecommendations.length === 0) {
        alert('推奨買い目がありません。先にレース予測を実行してください。');
        return;
    }

    // ブラウザ起動確認モーダルを表示
    showIpatLaunchConfirmModal();
}

/**
 * ブラウザ起動確認モーダルを表示
 * (旧 ipatVoteConfirmModal を流用)
 */
function showIpatLaunchConfirmModal() {
    const totalAmount = currentRecommendations.reduce((sum, rec) => sum + (rec.amount || 0), 0);
    const voteDetails = document.getElementById('voteDetails');

    if (voteDetails) {
        const rows = currentRecommendations.map(rec => `
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                <span>
                    <span class="badge badge-primary">${rec.type}</span> 
                    <b>${rec.umaban || rec.combination}</b>
                </span>
                <span>¥${rec.amount ? rec.amount.toLocaleString() : 0}</span>
            </div>
        `).join('');

        voteDetails.innerHTML = `
            <h4 style="margin-bottom: var(--space-md);">投票予定内容 (ブラウザへ転送)</h4>
            <div style="background: var(--bg-secondary); padding: var(--space-md); border-radius: 8px; max-height: 300px; overflow-y: auto;">
                ${rows}
                <div style="display: flex; justify-content: space-between; padding-top: 12px; margin-top: 8px; border-top: 2px solid var(--border-color); font-weight: bold;">
                    <span>合計</span>
                    <span style="color: var(--accent);">¥${totalAmount.toLocaleString()}</span>
                </div>
            </div>

        `;
    }

    // ボタンのテキスト変更
    const confirmBtn = document.getElementById('confirmVoteBtn');
    if (confirmBtn) {
        confirmBtn.textContent = 'ブラウザを起動して投票へ 🚀';
    }

    if (elements.ipatVoteConfirmModal) {
        elements.ipatVoteConfirmModal.classList.remove('hidden');
    }
}

/**
 * モーダルを閉じる
 */
function closeIpatVoteModal() {
    if (elements.ipatVoteConfirmModal) {
        elements.ipatVoteConfirmModal.classList.add('hidden');
    }
}

/**
 * ブラウザ起動処理 (旧 handleConfirmVote)
 */
async function handleConfirmVote() {
    const confirmBtn = document.getElementById('confirmVoteBtn');
    const originalText = confirmBtn.textContent;
    confirmBtn.textContent = '起動中... (数秒かかります)';
    confirmBtn.disabled = true;

    try {
        // Betsデータ整形
        const bets = currentRecommendations.map(rec => {
            // 単勝・複勝の場合は umaban を数値として使用
            // 馬連・ワイド等の場合は combination を文字列として使用
            let horseNo;
            if (rec.type === '単勝' || rec.type === '複勝') {
                // 単複の場合、umabanを数値化
                horseNo = parseInt(rec.umaban);
                if (isNaN(horseNo)) {
                    console.warn(`Invalid umaban for ${rec.type}: ${rec.umaban}`);
                    horseNo = rec.umaban; // フォールバック
                }
            } else {
                // 組み合わせ馬券の場合、combinationをそのまま使用（例: "1-2"）
                horseNo = rec.combination || rec.umaban;
            }

            return {
                horse_no: horseNo,
                type: BET_TYPE_CODES[rec.type],
                amount: rec.amount || 100
            };
        });

        console.log('Sending bets to backend:', bets);

        const response = await fetch(`${API_BASE}/api/ipat/launch_browser`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                race_id: currentRaceId,
                bets: bets
            })
        });

        const data = await response.json();

        if (data.success) {
            alert('✅ ブラウザを起動しました！\n\n開いたブラウザ上で投票手続きを完了させてください。');
            closeIpatVoteModal();
        } else {
            alert('❌ 起動エラー: ' + (data.error || '不明なエラーが発生しました'));
        }

    } catch (error) {
        console.error('Launch error:', error);
        alert('通信エラーが発生しました: ' + error.message);
    } finally {
        confirmBtn.textContent = originalText;
        confirmBtn.disabled = false;
    }
}

// 古い関数（削除済み）への参照が残っている場合のダミー（念のため）
function closeIpatLoginModal() { /* NOOP */ }

