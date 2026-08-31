<template>
    <Transition name="fade">
        <div class="physton-prompt-favorite" ref="favorite" v-show="isShow" @mouseenter="onMouseEnter"
             @mouseleave="onMouseLeave" @click.stop="">
            <!-- export / import actions (#330) -->
            <div class="popup-actions">
                <div class="popup-action-btn" @click="exportFavorites" :title="'Export favorites as JSON'">
                    <icon-svg name="copy"/> Export JSON
                </div>
                <label class="popup-action-btn" :title="'Import favorites from JSON'">
                    <icon-svg name="use"/> Import JSON
                    <input ref="importInput" type="file" accept=".json" style="display:none"
                           @change="onImportFileChange"/>
                </label>
                <div class="popup-action-btn" @click="exportFavoritesCSV" :title="'Export favorites as CSV (Excel)'"  >
                    <icon-svg name="copy"/> Export CSV
                </div>
                <label class="popup-action-btn" :title="'Import favorites from CSV'">
                    <icon-svg name="use"/> Import CSV
                    <input ref="importCSVInput" type="file" accept=".csv" style="display:none"
                           @change="onImportCSVFileChange"/>
                </label>
            </div>
            <div class="popup-tabs">
                <div v-for="(group) in favorites" :key="group.key"
                     :class="['popup-tab', group.key === favoriteKey ? 'active': '']" @click="onTabClick(group.key)">
                    <div class="tab-name">{{ getLang(group.name) }}</div>
                    <div class="tab-type">{{ getLang(group.type) }}</div>
                    <div class="tab-count">{{ group.list.length }}</div>
                </div>
            </div>
            <div class="popup-detail" ref="favoriteDetail" v-show="currentItem && currentItem.tags">
                <div class="popup-item-tags">
                    <template v-for="(tag, index) in currentItem.tags" :key="index">
                        <div v-if="tag.type && tag.type === 'wrap'" class="item-wrap"></div>
                        <div v-else class="item-tag">
                            <div class="item-tag-value">{{ tag.value }}</div>
                            <div class="item-tag-local-value">{{ tag.localValue }}</div>
                        </div>
                    </template>
                </div>
            </div>
            <div v-for="(group) in favorites" :key="group.key" :class="['popup-tab-content', group.key === favoriteKey ? 'active': '']">
                <div class="content-list" v-show="group.list.length > 0">
                    <div class="content-item" v-for="(item, index) in group.list" :key="item.id"
                         @mouseenter="onItemMouseEnter(index)" @mouseleave="onItemMouseLeave(index)">
                        <div class="item-header">
                            <div class="item-header-left">
                                <div class="item-header-index">{{ group.list.length - index }}</div>
                                <div class="item-header-time">{{ formatTime(item.time) }}</div>
                                <div class="item-header-name">
                                    <input class="header-name-input" :value="item.name"
                                           @keydown="onNameKeyDown(index, $event)"
                                           @change="onNameChange(index, $event)" :placeholder="getLang('unset_name')">
                                </div>
                            </div>
                            <div class="item-header-right">
                                <div class="header-btn-favorite hover-scale-140" @click="onFavoriteClick(index)"
                                     v-show="item.is_favorite" v-tooltip="getLang('remove_from_favorite')">
                                    <icon-svg name="favorite-yes"/>
                                </div>
                                <div class="header-btn-favorite hover-scale-140" @click="onFavoriteClick(index)"
                                     v-show="!item.is_favorite" v-tooltip="getLang('add_to_favorite')">
                                    <icon-svg name="favorite-no"/>
                                </div>
                                <div class="header-btn-copy hover-scale-140" @click="onCopyClick(index)"
                                     v-tooltip="getLang('copy_to_clipboard')">
                                    <icon-svg name="copy"/>
                                </div>
                                <div class="header-btn-use hover-scale-140" @click="onUseClick(index)"
                                     v-tooltip="getLang('use')">
                                    <icon-svg name="use"/>
                                </div>
                                <div class="header-btn-move-up hover-scale-140" @click="onMoveUpClick(index)"
                                     v-tooltip="getLang('move_up')">
                                    <icon-svg name="move-up"/>
                                </div>
                                <div class="header-btn-move-down hover-scale-140" @click="onMoveDownClick(index)"
                                     v-tooltip="getLang('move_down')">
                                    <icon-svg name="move-down"/>
                                </div>
                            </div>
                        </div>
                        <div class="item-prompt">{{ item.prompt }}</div>
                    </div>
                </div>
                <div class="content-empty" v-show="group.list.length === 0">
                    <icon-svg name="loading" v-if="loading"/>
                    <span v-else>{{ emptyMsg }}</span>
                </div>
            </div>
        </div>
    </Transition>
</template>
<script>
import common from "@/utils/common";

import LanguageMixin from "@/mixins/languageMixin";
import IconSvg from "@/components/iconSvg.vue";
import waitTick from '@/utils/waitTick';

export default {
    components: {IconSvg},
    props: {},
    mixins: [LanguageMixin],
    data() {
        return {
            favoriteKey: '',
            favorites: [
                {
                    'name': 'txt2img',
                    'type': 'prompt',
                    'key': 'txt2img',
                    'list': [],
                },
                {
                    'name': 'txt2img',
                    'type': 'negative_prompt',
                    'key': 'txt2img_neg',
                    'list': [],
                },
                {
                    'name': 'img2img',
                    'type': 'prompt',
                    'key': 'img2img',
                    'list': [],
                },
                {
                    'name': 'img2img',
                    'type': 'negative_prompt',
                    'key': 'img2img_neg',
                    'list': [],
                },
            ],
            isShow: false,
            loading: false,
            emptyMsg: '',
            mouseEnter: false,
            currentItem: {}
        }
    },
    emits: ['use'],
    mounted() {
        this.favorites.forEach(item => {
            waitTick.addWaitTick(() => this.getFavorites(item.key))
        })
    },
    methods: {
        formatTime(time) {
            return common.formatTime(time * 1000, false)
        },
        getFavorites(favoriteKey) {
            if (!favoriteKey) return
            let favoriteItem = this.favorites.find(item => item.key === favoriteKey)
            if (!favoriteItem) return
            this.loading = true
            return this.gradioAPI.getFavorites(favoriteKey).then(res => {
                if(res && res.length > 0){
                    // 倒序
                    res.reverse()
                    res.forEach(item => {
                        item.is_favorite = true
                    })
                    favoriteItem.list = res
                }
                window.phystonPromptfavorites = this.favorites
                this.emptyMsg = this.getLang('no_favorite')
                this.loading = false
            }).catch(err => {
                this.emptyMsg = this.getLang('get_favorite_error')
                this.loading = false
            })
        },
        show(favoriteKey, e) {
            if (!favoriteKey || !e) return
            this.favoriteKey = favoriteKey
            if (this.isShow) {
                this.isShow = false
                return
            }
            this.mouseEnter = false

            this.loading = true
            this.isShow = true
            this.$refs.favorite.style.top = (e.pageY + 2) + 'px'
            this.$refs.favorite.style.left = (e.pageX + 2) + 'px'

            this.getFavorites(this.favoriteKey)
            this.$nextTick(() => {
                // 如果当前窗口超出屏幕，就自动调整位置
                let rect = this.$refs.favorite.getBoundingClientRect()
                if (rect.right > window.innerWidth) {
                    this.$refs.favorite.style.left = (window.innerWidth - rect.width - 2) + 'px'
                }
            })

            // 如果n秒后鼠标还没进来，就隐藏
            setTimeout(() => {
                if (this.mouseEnter) return
                this.hide()
            }, 3000)


        },
        hide() {
            this.mouseEnter = false
            this.isShow = false
        },
        onMouseEnter() {
            this.mouseEnter = true
        },
        onMouseLeave(e) {
            if (!e.relatedTarget) return // 微软输入法BUG
            this.hide()
        },
        onTabClick(key) {
            this.favoriteKey = key
            this.getFavorites(this.favoriteKey)
        },
        onFavoriteClick(index) {
            let group = this.favorites.find(item => item.key === this.favoriteKey)
            if (!group) return
            let favorite = group.list[index]
            if (!favorite.is_favorite) {
                this.gradioAPI.doFavorite(this.favoriteKey, favorite.id).then(res => {
                    if (res) {
                        favorite.is_favorite = true
                        window.phystonPromptfavorites = this.favorites
                    }
                })
            } else {
                this.gradioAPI.unFavorite(this.favoriteKey, favorite.id).then(res => {
                    if (res) {
                        favorite.is_favorite = false
                        window.phystonPromptfavorites = this.favorites
                    }
                })
            }
        },
        onCopyClick(index) {
            let group = this.favorites.find(item => item.key === this.favoriteKey)
            if (!group) return
            let favorite = group.list[index]
            this.$copyText(favorite.prompt).then(() => {
                this.$toastr.success("success!")
            }).catch(() => {
                this.$toastr.error("error!")
            })
        },
        onNameKeyDown(index, e) {
            if (e.keyCode === 13) {
                // 离开焦点
                e.target.blur()
            }
        },
        onNameChange(index, e) {
            let group = this.favorites.find(item => item.key === this.favoriteKey)
            if (!group) return
            let favorite = group.list[index]
            const value = e.target.value
            this.gradioAPI.setFavoriteName(this.favoriteKey, favorite.id, value).then(res => {
                if (res) {
                    favorite.name = value
                    window.phystonPromptfavorites = this.favorites
                } else {
                    e.target.value = favorite.name
                }
            }).catch(err => {
                e.target.value = favorite.name
            })
        },
        onItemMouseEnter(index) {
            let group = this.favorites.find(item => item.key === this.favoriteKey)
            if (!group) return
            this.currentItem = group.list[index]

            this.$nextTick(() => {
                // 判断 favoriteDetail 是否超出屏幕
                let rect = this.$refs.favoriteDetail.getBoundingClientRect()
                if (rect.right > window.innerWidth) {
                    this.$refs.favoriteDetail.style.left = (0 - rect.width - 2) + 'px'
                }
            })
        },
        onItemMouseLeave(index) {
            this.currentItem = {}
        },
        onUseClick(index) {
            let group = this.favorites.find(item => item.key === this.favoriteKey)
            if (!group) return
            this.hide()
            this.$emit('use', group.list[index])
        },
        onMoveUpClick(index) {
            let group = this.favorites.find(item => item.key === this.favoriteKey)
            if (!group) return
            let favorite = group.list[index]
            if (index === 0) return
            this.gradioAPI.moveDownFavorite(this.favoriteKey, favorite.id).then(res => {
            // this.gradioAPI.moveUpFavorite(this.favoriteKey, favorite.id).then(res => {
                if (res) {
                    group.list.splice(index, 1)
                    group.list.splice(index - 1, 0, favorite)
                    window.phystonPromptfavorites = this.favorites
                }
            })
        },
        onMoveDownClick(index) {
            let group = this.favorites.find(item => item.key === this.favoriteKey)
            if (!group) return
            let favorite = group.list[index]
            if (index === group.list.length - 1) return
            this.gradioAPI.moveUpFavorite(this.favoriteKey, favorite.id).then(res => {
            // this.gradioAPI.moveDownFavorite(this.favoriteKey, favorite.id).then(res => {
                if (res) {
                    group.list.splice(index, 1)
                    group.list.splice(index + 1, 0, favorite)
                    window.phystonPromptfavorites = this.favorites
                }
            })
        },

        // ---- Export / Import CSV (#csv) ------------------------------------
        /**
         * Escape a single value for RFC-4180 CSV.
         * Fields containing comma, double-quote or newline are wrapped in quotes;
         * internal double-quotes are doubled.
         */
        csvEscape(val) {
            const s = (val === null || val === undefined) ? '' : String(val)
            if (s.includes(',') || s.includes('"') || s.includes('"') || s.includes('\n') || s.includes('\r')) {
                return '"' + s.replace(/"/g, '""') + '"'
            }
            return s
        },
        /**
         * Download all favorite groups as a single CSV file.
         * Columns: group, name, prompt, date
         * UTF-8 BOM prefix so Excel auto-detects encoding on Windows.
         */
        exportFavoritesCSV() {
            const rows = [['group', 'name', 'prompt', 'date']]
            this.favorites.forEach(group => {
                group.list.forEach(item => {
                    rows.push([
                        group.key,
                        item.name || '',
                        item.prompt || '',
                        this.formatTime(item.time),
                    ])
                })
            })
            const csv = '\uFEFF' + rows.map(r => r.map(f => this.csvEscape(f)).join(',')).join('\n')
            const blob = new Blob([csv], {type: 'text/csv;charset=utf-8'})
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = 'paio-neo-favorites.csv'
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            URL.revokeObjectURL(url)
        },
        /**
         * Parse an RFC-4180 CSV string into an array of rows (each row = array of strings).
         * Handles BOM, quoted fields, escaped quotes and CRLF/LF line endings.
         */
        parseCSV(text) {
            const src = text.charCodeAt(0) === 0xFEFF ? text.slice(1) : text
            const rows = []
            let i = 0
            const n = src.length
            while (i < n) {
                const row = []
                while (i < n) {
                    let field = ''
                    if (src[i] === '"') {
                        i++ // opening quote
                        while (i < n) {
                            if (src[i] === '"') {
                                if (i + 1 < n && src[i + 1] === '"') {
                                    field += '"'; i += 2
                                } else {
                                    i++; break // closing quote
                                }
                            } else {
                                field += src[i++]
                            }
                        }
                    } else {
                        while (i < n && src[i] !== ',' && src[i] !== '\n' && src[i] !== '\r') {
                            field += src[i++]
                        }
                    }
                    row.push(field)
                    if (i < n && src[i] === ',') { i++; continue }
                    break
                }
                if (i < n && src[i] === '\r') i++
                if (i < n && src[i] === '\n') i++
                if (!(row.length === 1 && row[0] === '')) rows.push(row)
            }
            return rows
        },
        /** Read the selected CSV file and replace storage for each group present */
        async onImportCSVFileChange(e) {
            const file = e.target.files[0]
            if (!file) return
            try {
                const text = await file.text()
                const rows = this.parseCSV(text)
                if (rows.length < 2) throw new Error('Empty file')
                const header = rows[0].map(h => h.trim().toLowerCase())
                const gi = header.indexOf('group')
                const ni = header.indexOf('name')
                const pi = header.indexOf('prompt')
                if (gi === -1 || pi === -1) throw new Error('Missing required columns: group, prompt')
                // Group data rows by group key
                const byGroup = {}
                const validKeys = this.favorites.map(g => g.key)
                rows.slice(1).forEach((row, idx) => {
                    const key = row[gi] ? row[gi].trim() : ''
                    if (!validKeys.includes(key)) return
                    if (!byGroup[key]) byGroup[key] = []
                    byGroup[key].push({
                        id: Date.now() + idx,
                        prompt: row[pi] || '',
                        name: ni !== -1 ? (row[ni] || '') : '',
                        time: Math.floor(Date.now() / 1000),
                        is_favorite: true,
                        tags: [],
                    })
                })
                if (Object.keys(byGroup).length === 0) throw new Error('No valid group rows found')
                for (const key of Object.keys(byGroup)) {
                    await this.gradioAPI.setData('favorite.' + key, byGroup[key])
                }
                for (const group of this.favorites) {
                    await this.getFavorites(group.key)
                }
                this.$toastr.success('Favorites imported from CSV!')
            } catch (err) {
                this.$toastr.error('CSV import failed: ' + err.message)
            } finally {
                e.target.value = ''
            }
        },

        // ---- Export / Import JSON (#330) ------------------------------------
        /**
         * Download all favorite groups as a single JSON file.
         * Format: { [groupKey]: [items...], ... }
         */
        exportFavorites() {
            const data = {}
            this.favorites.forEach(group => {
                data[group.key] = group.list
            })
            const json = JSON.stringify(data, null, 2)
            const blob = new Blob([json], {type: 'application/json'})
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = 'paio-neo-favorites.json'
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            URL.revokeObjectURL(url)
        },
        /** Trigger the hidden file input */
        importFavorites() {
            this.$refs.importInput.click()
        },
        /** Read the selected JSON file and merge into storage */
        async onImportFileChange(e) {
            const file = e.target.files[0]
            if (!file) return
            try {
                const text = await file.text()
                const data = JSON.parse(text)
                if (typeof data !== 'object' || Array.isArray(data)) throw new Error('Invalid format')
                const validKeys = this.favorites.map(g => g.key)
                for (const key of Object.keys(data)) {
                    if (!validKeys.includes(key)) continue
                    const list = data[key]
                    if (!Array.isArray(list)) continue
                    await this.gradioAPI.setData('favorite.' + key, list)
                }
                // Reload all groups from storage
                for (const group of this.favorites) {
                    await this.getFavorites(group.key)
                }
                this.$toastr.success('Favorites imported!')
            } catch (err) {
                this.$toastr.error('Import failed: ' + err.message)
            } finally {
                e.target.value = ''
            }
        },
    }
}
</script>
