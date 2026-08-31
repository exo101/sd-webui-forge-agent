<template>
    <Transition name="fadeDown">
        <div class="physton-prompt-quality-presets" v-if="isOpen" @click="onBackdropClick">
            <div class="qp-main" @click.stop>

                <!-- header -->
                <div class="qp-header">
                    <span class="qp-title">Quality Presets</span>
                    <div class="qp-close" @click="close"><icon-svg name="close"/></div>
                </div>

                <!-- tabs -->
                <div class="qp-tabs">
                    <div :class="['qp-tab', activeTab === 'templates' && 'active']"
                         @click="switchTab('templates')">Templates</div>
                    <div :class="['qp-tab', activeTab === 'checkpoints' && 'active']"
                         @click="switchTab('checkpoints')">Checkpoints</div>
                    <div :class="['qp-tab', activeTab === 'custom' && 'active']"
                         @click="switchTab('custom')">Custom</div>
                </div>

                <!-- ================================================ TEMPLATES -->
                <div class="qp-panel" v-if="activeTab === 'templates'">
                    <div class="qp-panel-hint">
                        Enable families that apply to your models. Tags are injected when a checkpoint
                        is matched to that family. You can override the default tags per family.
                    </div>

                    <div class="qp-family-card"
                         v-for="(tpl, family) in builtinTemplates"
                         :key="family">

                        <div class="qp-family-header">
                            <label class="qp-family-toggle">
                                <input type="checkbox"
                                       :checked="builtinEnabled[family] !== false"
                                       @change="toggleFamily(family, $event.target.checked)"/>
                                <span class="qp-family-name">{{ family }}</span>
                            </label>
                            <div class="qp-btn qp-btn-xs"
                                 @click="toggleEditFamily(family)">
                                {{ editingFamily === family ? 'Done' : 'Edit' }}
                            </div>
                        </div>

                        <!-- read-only view -->
                        <template v-if="editingFamily !== family">
                            <div class="qp-family-tags" v-if="resolvedTags(family).pos.length">
                                <span class="qp-chip"
                                      v-for="t in resolvedTags(family).pos" :key="t">{{ t }}</span>
                            </div>
                            <div class="qp-family-tags qp-family-tags-neg"
                                 v-if="resolvedTags(family).neg.length">
                                <span class="qp-chip qp-chip-neg"
                                      v-for="t in resolvedTags(family).neg" :key="t">{{ t }}</span>
                            </div>
                            <div class="qp-override-badge"
                                 v-if="builtinOverrides[family]">overridden</div>
                        </template>

                        <!-- edit form -->
                        <div class="qp-family-edit" v-else>
                            <div class="qp-form-row">
                                <label>Positive tags (comma-separated)</label>
                                <input type="text"
                                       v-model="editForms[family].pos"
                                       placeholder="tag1, tag2, ..."/>
                            </div>
                            <div class="qp-form-row">
                                <label>Negative tags (comma-separated)</label>
                                <input type="text"
                                       v-model="editForms[family].neg"
                                       placeholder="worst quality, ..."/>
                            </div>
                            <div class="qp-form-actions">
                                <div class="qp-btn qp-btn-xs" @click="saveEditFamily(family)">Apply</div>
                                <div class="qp-btn qp-btn-xs qp-btn-danger"
                                     @click="resetFamily(family)"
                                     v-if="builtinOverrides[family]">Reset to default</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- ================================================ CHECKPOINTS -->
                <div class="qp-panel" v-if="activeTab === 'checkpoints'">
                    <div class="qp-panel-actions">
                        <div class="qp-btn" @click="loadCheckpoints">
                            <icon-svg v-if="loadingCheckpoints" name="loading"/>
                            <template v-else>Refresh</template>
                        </div>
                        <div class="qp-btn" @click="scanAll" :class="{disabled: scanningAll}">
                            <icon-svg v-if="scanningAll" name="loading"/>
                            <template v-else>Scan All via CivitAI</template>
                        </div>
                    </div>

                    <div class="qp-checkpoint-list">
                        <div class="qp-ckpt-item"
                             v-for="(ckpt, idx) in checkpoints" :key="ckpt.filepath">
                            <div class="qp-ckpt-name">{{ ckpt.title || ckpt.filename }}</div>
                            <div class="qp-ckpt-row">
                                <span class="qp-ckpt-file">{{ ckpt.filename }}</span>
                                <span v-if="ckpt.base_model" class="source-badge source-civitai">
                                    {{ ckpt.base_model }}
                                </span>
                                <span v-else-if="scanningIndex === idx" class="source-badge source-builtin">
                                    scanning...
                                </span>
                                <span v-else class="source-badge source-none">unscanned</span>
                                <div class="qp-btn qp-btn-xs"
                                     @click="scanOne(ckpt, idx)"
                                     :class="{disabled: scanningIndex === idx}">
                                    Detect
                                </div>
                            </div>
                        </div>
                        <div class="qp-empty" v-if="!loadingCheckpoints && !checkpoints.length">
                            Click Refresh to load installed checkpoints.
                        </div>
                    </div>
                </div>

                <!-- ================================================ CUSTOM -->
                <div class="qp-panel" v-if="activeTab === 'custom'">
                    <div class="qp-panel-hint">
                        Custom presets are matched by filename substring. They take priority over
                        built-in templates.
                    </div>

                    <div class="qp-section-header">
                        Custom Presets
                        <div class="qp-btn qp-btn-xs"
                             @click="showAddForm = !showAddForm">+ Add</div>
                    </div>

                    <div class="qp-add-form" v-if="showAddForm">
                        <div class="qp-form-row">
                            <label>Name</label>
                            <input type="text" v-model="newPreset.name" placeholder="My Pony Models"/>
                        </div>
                        <div class="qp-form-row">
                            <label>Filename substrings (comma-separated)</label>
                            <input type="text" v-model="newPreset.match_substr_str"
                                   placeholder="pony, pdxl"/>
                        </div>
                        <div class="qp-form-row">
                            <label>Positive prefix tags</label>
                            <input type="text" v-model="newPreset.positive_prefix_str"
                                   placeholder="score_9, score_8_up"/>
                        </div>
                        <div class="qp-form-row">
                            <label>Negative prefix tags</label>
                            <input type="text" v-model="newPreset.negative_prefix_str"
                                   placeholder="worst quality"/>
                        </div>
                        <div class="qp-form-row qp-form-inline">
                            <label>Auto-insert on load</label>
                            <input type="checkbox" v-model="newPreset.auto_insert"/>
                        </div>
                        <div class="qp-form-actions">
                            <div class="qp-btn qp-btn-xs" @click="onAddPreset">Save</div>
                            <div class="qp-btn qp-btn-xs qp-btn-cancel"
                                 @click="showAddForm = false">Cancel</div>
                        </div>
                    </div>

                    <div class="qp-preset-list">
                        <div class="qp-preset-item"
                             v-for="(preset, idx) in userPresets" :key="idx">
                            <div class="qp-preset-header">
                                <span class="qp-preset-name">{{ preset.name }}</span>
                                <span class="qp-auto-badge" v-if="preset.auto_insert">auto</span>
                                <div class="qp-preset-del" @click="onDeletePreset(idx)">
                                    <icon-svg name="remove"/>
                                </div>
                            </div>
                            <div class="qp-preset-meta"
                                 v-if="preset.match_substr && preset.match_substr.length">
                                match: {{ preset.match_substr.join(', ') }}
                            </div>
                            <div class="qp-tags qp-tags-sm"
                                 v-if="preset.positive_prefix && preset.positive_prefix.length">
                                <span class="qp-chip"
                                      v-for="t in preset.positive_prefix" :key="t">{{ t }}</span>
                            </div>
                        </div>
                        <div class="qp-empty" v-if="!userPresets.length">No custom presets yet.</div>
                    </div>
                </div>

            </div>
        </div>
    </Transition>
</template>

<script>
import LanguageMixin from "@/mixins/languageMixin";
import IconSvg from "@/components/iconSvg.vue";

export default {
    name: 'QualityPresets',
    components: {IconSvg},
    mixins: [LanguageMixin],
    emits: ['insert-positive', 'insert-negative'],
    data() {
        return {
            isOpen: false,
            activeTab: 'templates',

            // Templates tab
            builtinTemplates: {},
            builtinEnabled: {},
            builtinOverrides: {},
            editingFamily: null,
            editForms: {},

            // Checkpoints tab
            checkpoints: [],
            loadingCheckpoints: false,
            scanningIndex: null,
            scanningAll: false,

            // Custom tab
            userPresets: [],
            showAddForm: false,
            newPreset: {
                name: '',
                match_substr_str: '',
                positive_prefix_str: '',
                negative_prefix_str: '',
                auto_insert: false,
            },
        }
    },
    methods: {
        // ---- lifecycle -------------------------------------------------------
        open() {
            this.isOpen = true
            this.editingFamily = null
            this.showAddForm = false
            this._resetNewPreset()
            this._loadAll()
        },
        close() {
            this.isOpen = false
        },
        onBackdropClick() {
            this.close()
        },
        switchTab(name) {
            this.activeTab = name
            if (name === 'checkpoints' && !this.checkpoints.length) {
                this.loadCheckpoints()
            }
        },

        // ---- data loading ----------------------------------------------------
        _loadAll() {
            Promise.all([
                this.gradioAPI.getQualityPresets(),
                this.gradioAPI.getBuiltinTemplates(),
            ]).then(([presets, templates]) => {
                if (presets) {
                    this.userPresets = presets.presets || []
                    this.builtinEnabled = presets.builtin_enabled || {}
                    this.builtinOverrides = presets.builtin_overrides || {}
                }
                if (templates) {
                    this.builtinTemplates = templates
                    // Seed editForms for each family
                    const forms = {}
                    for (const family of Object.keys(templates)) {
                        const res = this.resolvedTags(family)
                        forms[family] = {
                            pos: res.pos.join(', '),
                            neg: res.neg.join(', '),
                        }
                    }
                    this.editForms = forms
                }
            }).catch(() => {})
        },
        _saveAll() {
            return this.gradioAPI.saveQualityPresets({
                builtin_enabled: this.builtinEnabled,
                builtin_overrides: this.builtinOverrides,
                presets: this.userPresets,
            })
        },

        // ---- Templates tab ---------------------------------------------------
        resolvedTags(family) {
            const override = this.builtinOverrides[family]
            const base     = this.builtinTemplates[family] || {}
            return {
                pos: (override ? override.positive_prefix : base.positive_prefix) || [],
                neg: (override ? override.negative_prefix : base.negative_prefix) || [],
            }
        },
        toggleFamily(family, enabled) {
            this.builtinEnabled = {...this.builtinEnabled, [family]: enabled}
            this._saveAll()
        },
        toggleEditFamily(family) {
            if (this.editingFamily === family) {
                this.editingFamily = null
                return
            }
            const res = this.resolvedTags(family)
            this.editForms = {
                ...this.editForms,
                [family]: {
                    pos: res.pos.join(', '),
                    neg: res.neg.join(', '),
                },
            }
            this.editingFamily = family
        },
        saveEditFamily(family) {
            const form = this.editForms[family] || {}
            const pos = (form.pos || '').split(',').map(s => s.trim()).filter(Boolean)
            const neg = (form.neg || '').split(',').map(s => s.trim()).filter(Boolean)
            this.builtinOverrides = {
                ...this.builtinOverrides,
                [family]: {positive_prefix: pos, negative_prefix: neg},
            }
            this.editingFamily = null
            this._saveAll()
        },
        resetFamily(family) {
            const overrides = {...this.builtinOverrides}
            delete overrides[family]
            this.builtinOverrides = overrides
            const base = this.builtinTemplates[family] || {}
            this.editForms = {
                ...this.editForms,
                [family]: {
                    pos: (base.positive_prefix || []).join(', '),
                    neg: (base.negative_prefix || []).join(', '),
                },
            }
            this.editingFamily = null
            this._saveAll()
        },

        // ---- Checkpoints tab -------------------------------------------------
        loadCheckpoints() {
            if (this.loadingCheckpoints) return
            this.loadingCheckpoints = true
            this.gradioAPI.getInstalledCheckpoints().then(list => {
                this.checkpoints = list || []
                this.loadingCheckpoints = false
            }).catch(() => {
                this.loadingCheckpoints = false
            })
        },
        scanOne(ckpt, idx) {
            if (this.scanningIndex === idx) return
            this.scanningIndex = idx
            this.gradioAPI.scanCheckpoint(ckpt.filepath).then(result => {
                if (result && result.base_model !== undefined) {
                    this.checkpoints[idx] = {
                        ...this.checkpoints[idx],
                        base_model: result.base_model,
                    }
                }
                this.scanningIndex = null
            }).catch(() => {
                this.scanningIndex = null
            })
        },
        async scanAll() {
            if (this.scanningAll) return
            this.scanningAll = true
            for (let i = 0; i < this.checkpoints.length; i++) {
                const ckpt = this.checkpoints[i]
                if (ckpt.base_model) continue
                this.scanningIndex = i
                try {
                    const result = await this.gradioAPI.scanCheckpoint(ckpt.filepath)
                    if (result && result.base_model !== undefined) {
                        this.checkpoints[i] = {...ckpt, base_model: result.base_model}
                    }
                } catch (_) { /* skip */ }
            }
            this.scanningIndex = null
            this.scanningAll = false
        },

        // ---- Custom tab ------------------------------------------------------
        _resetNewPreset() {
            this.newPreset = {
                name: '',
                match_substr_str: '',
                positive_prefix_str: '',
                negative_prefix_str: '',
                auto_insert: false,
            }
        },
        onAddPreset() {
            if (!this.newPreset.name.trim()) return
            this.userPresets.push({
                name: this.newPreset.name.trim(),
                match_exact: [],
                match_substr: this.newPreset.match_substr_str
                    .split(',').map(s => s.trim()).filter(Boolean),
                positive_prefix: this.newPreset.positive_prefix_str
                    .split(',').map(s => s.trim()).filter(Boolean),
                negative_prefix: this.newPreset.negative_prefix_str
                    .split(',').map(s => s.trim()).filter(Boolean),
                positive_embeddings: [],
                negative_embeddings: [],
                auto_insert: this.newPreset.auto_insert,
            })
            this._saveAll()
            this.showAddForm = false
            this._resetNewPreset()
        },
        onDeletePreset(idx) {
            this.userPresets.splice(idx, 1)
            this._saveAll()
        },
    },
}
</script>

<style scoped>
/* ---- overlay ---- */
.physton-prompt-quality-presets {
    position: fixed;
    inset: 0;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, .45);
}

/* ---- main panel ---- */
.qp-main {
    position: relative;
    width: 560px;
    max-width: 96vw;
    max-height: 90vh;
    overflow-y: auto;
    background: var(--background-fill-primary, #1a1a1a);
    border-radius: 8px;
    padding: 18px 22px 22px;
    box-sizing: border-box;
    color: var(--body-text-color, #eee);
    font-size: 13px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* ---- header ---- */
.qp-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.qp-title {
    font-size: 15px;
    font-weight: 600;
}
.qp-close {
    cursor: pointer;
    opacity: .55;
    width: 16px;
    height: 16px;
    transition: opacity .15s;
}
.qp-close:hover { opacity: 1; }

/* ---- tabs ---- */
.qp-tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border-color-primary, #333);
}
.qp-tab {
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    opacity: .55;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: opacity .15s, border-color .15s;
    user-select: none;
}
.qp-tab.active {
    opacity: 1;
    border-bottom-color: var(--primary-500, #4a90e2);
}
.qp-tab:hover { opacity: .8; }

/* ---- shared panel ---- */
.qp-panel {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.qp-panel-hint {
    font-size: 11px;
    opacity: .5;
    line-height: 1.5;
}
.qp-panel-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

/* ---- buttons ---- */
.qp-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    padding: 5px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    background: var(--button-secondary-background-fill, #3a3a3a);
    color: var(--button-secondary-text-color, #eee);
    border: 1px solid var(--border-color-primary, #555);
    transition: opacity .15s;
    white-space: nowrap;
    user-select: none;
}
.qp-btn:hover { opacity: .8; }
.qp-btn.disabled { opacity: .4; cursor: default; pointer-events: none; }
.qp-btn-xs { padding: 3px 8px; font-size: 11px; }
.qp-btn-danger { background: #4a2020; border-color: #7a4a4a; }
.qp-btn-cancel { opacity: .6; }

/* ---- family cards (Templates tab) ---- */
.qp-family-card {
    background: var(--background-fill-secondary, #1e1e1e);
    border: 1px solid var(--border-color-primary, #333);
    border-radius: 6px;
    padding: 10px 12px;
}
.qp-family-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.qp-family-toggle {
    display: flex;
    align-items: center;
    gap: 7px;
    cursor: pointer;
    user-select: none;
}
.qp-family-toggle input[type="checkbox"] {
    width: 14px;
    height: 14px;
    cursor: pointer;
}
.qp-family-name { font-weight: 600; font-size: 13px; }
.qp-family-tags { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 3px; }
.qp-family-tags-neg .qp-chip { background: #4a2a2a; color: #d08080; }
.qp-override-badge {
    display: inline-block;
    font-size: 10px;
    background: #2a3a1a;
    color: #8add6a;
    border-radius: 8px;
    padding: 1px 6px;
    margin-top: 4px;
}

/* ---- family edit form ---- */
.qp-family-edit { margin-top: 8px; }
.qp-form-row { margin-bottom: 7px; }
.qp-form-row label {
    display: block;
    font-size: 11px;
    opacity: .65;
    margin-bottom: 3px;
}
.qp-form-row input[type="text"] {
    width: 100%;
    box-sizing: border-box;
    background: var(--input-background-fill, #2a2a2a);
    border: 1px solid var(--border-color-primary, #444);
    border-radius: 4px;
    color: inherit;
    padding: 4px 7px;
    font-size: 12px;
    outline: none;
}
.qp-form-inline { display: flex; align-items: center; gap: 8px; }
.qp-form-inline label { margin: 0; }
.qp-form-actions { display: flex; gap: 6px; margin-top: 8px; }

/* ---- checkpoints tab ---- */
.qp-checkpoint-list {
    display: flex;
    flex-direction: column;
    gap: 5px;
    max-height: 400px;
    overflow-y: auto;
}
.qp-ckpt-item {
    background: var(--background-fill-secondary, #1e1e1e);
    border: 1px solid var(--border-color-primary, #333);
    border-radius: 5px;
    padding: 7px 10px;
}
.qp-ckpt-name { font-weight: 500; font-size: 12px; margin-bottom: 4px; }
.qp-ckpt-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}
.qp-ckpt-file {
    font-size: 11px;
    opacity: .45;
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ---- common chips / badges ---- */
.qp-chip {
    background: #2a4a2a;
    color: #8add6a;
    border-radius: 3px;
    padding: 2px 7px;
    font-size: 11px;
}
.qp-chip-neg { background: #4a2a2a; color: #d08080; }
.qp-tags { display: flex; flex-wrap: wrap; gap: 4px; }
.qp-tags-sm .qp-chip { font-size: 10px; padding: 1px 5px; }

.source-badge {
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 10px;
    font-weight: 600;
    text-transform: uppercase;
    white-space: nowrap;
}
.source-civitai  { background: #1a3a5a; color: #6ab4ff; }
.source-preset   { background: #2a3a1a; color: #8add6a; }
.source-builtin  { background: #3a3a1a; color: #e0d060; }
.source-none     { background: #3a1a1a; color: #d06060; }

/* ---- custom tab ---- */
.qp-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 600;
    opacity: .7;
    text-transform: uppercase;
    letter-spacing: .05em;
    border-bottom: 1px solid var(--border-color-primary, #333);
    padding-bottom: 5px;
    margin-top: 4px;
}
.qp-add-form {
    background: var(--background-fill-secondary, #222);
    border: 1px solid var(--border-color-primary, #333);
    border-radius: 6px;
    padding: 12px;
}
.qp-preset-list { display: flex; flex-direction: column; gap: 6px; }
.qp-preset-item {
    background: var(--background-fill-secondary, #1e1e1e);
    border: 1px solid var(--border-color-primary, #333);
    border-radius: 5px;
    padding: 8px 10px;
}
.qp-preset-header { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.qp-preset-name { font-weight: 600; flex: 1; }
.qp-auto-badge {
    font-size: 10px;
    background: #1a3a5a;
    color: #6ab4ff;
    border-radius: 8px;
    padding: 1px 6px;
}
.qp-preset-del {
    cursor: pointer;
    opacity: .4;
    width: 14px;
    height: 14px;
    transition: opacity .15s;
}
.qp-preset-del:hover { opacity: .9; }
.qp-preset-meta { font-size: 11px; opacity: .5; margin-bottom: 4px; }

.qp-empty {
    font-size: 12px;
    opacity: .45;
    font-style: italic;
    padding: 6px 0;
}
</style>