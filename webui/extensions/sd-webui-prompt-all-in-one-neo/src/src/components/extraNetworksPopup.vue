<template>
    <div class="physton-prompt-extra-networks-popup" ref="extraNetworks"
         :data-from="from"
         @mouseenter.stop="onMouseEnter"
         @mousemove.stop="onMouseMove"
         @mouseleave.stop="onMouseLeave"
         v-show="isShow"
         :style="style">
        <div class="popup-main">
            <img class="info-preview" v-if="data.preview" :src="data.preview" :style="previewStyle" @load="onPreviewLoad"/>
            <div class="info-raws">
                <div class="info-raw">
                    <div class="raw-name">{{ getLang('model_name') }}</div>
                    <div class="raw-value" @click="copy(data.name)">{{ data.name }}</div>
                </div>
                <div class="info-raw" v-if="data.output_name">
                    <div class="raw-name">{{ getLang('output_name') }}</div>
                    <div class="raw-value" @click="copy(data.output_name)">{{ data.output_name }}</div>
                </div>
                <div class="info-raw" v-if="data.basename">
                    <div class="raw-name">{{ getLang('filename') }}</div>
                    <div class="raw-value" @click="copy(data.basename)">{{ data.basename }}</div>
                </div>
                <div class="info-raw" v-if="data.dirname">
                    <div class="raw-name">{{ getLang('filepath') }}</div>
                    <div class="raw-value" @click="copy(data.dirname)">{{ data.dirname }}</div>
                </div>
                <div class="info-raw" v-if="data.trainedWords && data.trainedWords.length">
                    <div class="raw-name">{{ getLang('trained_words') }}</div>
                    <div class="raw-value">
                        <div class="raw-words" v-for="(value) in data.trainedWords" :key="value" @click="onUseKeywordsClick(value)">{{ value }}</div>
                    </div>
                </div>
                <div class="info-raw" v-if="data.description">
                    <div class="raw-name">{{ getLang('description') }}</div>
                    <div class="raw-value" v-tooltip="data.description" @click="copy(data.description)"><div class="raw-description">{{ data.description }}</div></div>
                </div>
                <div class="info-raw" v-if="data.civitaiUrl">
                    <div class="raw-name">Civitai</div>
                    <div class="raw-value"><a :href="data.civitaiUrl" target="_blank" class="raw-link">{{ data.civitaiUrl }}</a></div>
                </div>
                <!--<div class="info-btns">
                    <div class="info-btn hover-scale-120" v-if="data.modelId" @click="onOpenCivitaiClick">{{ getLang('open_civitai') }}</div>
                    <div class="info-btn hover-scale-120" v-if="data.trainedWords && data.trainedWords.length" @click="onUseKeywordsClick">{{ getLang('use_keywords') }}</div>
                </div>-->
            </div>
        </div>
    </div>
</template>
<script>
import LanguageMixin from "@/mixins/languageMixin";
import IconSvg from "@/components/iconSvg.vue";
import common from "@/utils/common";
import globals from "../../globals";

const TYPE_LORA = 'lora'
const TYPE_LYCO = 'lyco'
const TYPE_EMBEDDING = 'embedding'
export default {
    name: 'ExtraNetworksPopup',
    components: {IconSvg},
    mixins: [LanguageMixin],
    props: {
        extraNetworks: {
            type: Array,
            default: () => [],
        },
    },
    data() {
        return {
            isShow: false,
            e: null,
            name: '',
            useCallback: null,
            type: '',
            from: 'tags',
            data: {},
            style: {},
            previewStyle: {},
            mouseIn: false,
            eMouseIn: false,
        }
    },
    computed: {},
    mounted() {
        // Keep the popup anchored to the hovered tag when Forge/Gradio uses
        // a scrollable container or the browser viewport changes.
        this._positionHandler = () => {
            if (this.isShow) this.updatePosition()
        }
        window.addEventListener('resize', this._positionHandler)
        window.addEventListener('scroll', this._positionHandler, true)
    },
    beforeUnmount() {
        if (this._positionHandler) {
            window.removeEventListener('resize', this._positionHandler)
            window.removeEventListener('scroll', this._positionHandler, true)
        }
    },
    methods: {
        show(e, name, useCallback, showCheckpoints = false, from = 'tags') {
            this.mouseIn = false
            this.eMouseIn = true
            this.e = e
            this.name = name
            this.useCallback = useCallback
            this.type = ''
            this.from = from
            this.data = {}
            this.style = {}
            this.previewStyle = {}
            let data
            name = name.toLowerCase()
            for (let extraNetwork of this.extraNetworks) {
                if (extraNetwork.name === 'textual inversion') {
                    for (let item of extraNetwork.items) {
                        if (item.name.toLowerCase() === name) {
                            this.type = TYPE_EMBEDDING
                            data = item
                            break
                        }
                    }
                } else if (extraNetwork.name === 'lora' || extraNetwork.name === 'lycoris') {
                    for (let item of extraNetwork.items) {
                        const itemBasename = item.name.replace(/\\/g, '/').split('/').pop()
                        if (
                            item.name.toLowerCase() === name ||
                            (item.output_name && item.output_name.toLowerCase() === name) ||
                            (itemBasename && itemBasename.toLowerCase() === name)
                        ) {
                            this.type = extraNetwork.name === 'lora' ? TYPE_LORA : TYPE_LYCO
                            data = item
                            break
                        }
                    }
                } else if (showCheckpoints && extraNetwork.name === 'checkpoints') {
                    for (let item of extraNetwork.items) {
                        if (item.name.toLowerCase() === name) {
                            this.type = 'checkpoints'
                            data = item
                            break
                        }
                    }
                }
            }
            if (!this.type) return this.isShow = false
            this.isShow = true

            this.data.name = data.civitai_info && data.civitai_info.model && data.civitai_info.model.name ? data.civitai_info.model.name : data.name
            this.data.output_name = data.output_name && data.output_name !== this.data.name ? data.output_name : ''
            this.data.basename = data.basename || ''
            this.data.dirname = data.dirname || ''
            this.data.description = data.civitai_info && data.civitai_info.description ? data.civitai_info.description : (data.description || '')
            this.data.description = this.data.description.replace(/<[^>]+>/g, '')
            this.data.trainedWords = data.civitai_info && data.civitai_info.trainedWords ? data.civitai_info.trainedWords : []
            this.data.preview = data.preview || (data.civitai_info && data.civitai_info.images && data.civitai_info.images.length ? data.civitai_info.images[0] : '')
            this.data.modelId = data.civitai_info && data.civitai_info.modelId ? data.civitai_info.modelId : ''
            this.data.civitaiUrl = this.data.modelId ? this.getCivitaiUrl(this.data.modelId) : ''
            this.$nextTick(() => {
                if (this.data.preview) {
                    this.previewStyle = {
                        width: 'auto',
                        height: this.$refs.extraNetworks.offsetHeight + 'px',
                    }
                }
                this.$nextTick(() => {
                    this.updatePosition()
                })
            })
        },
        updatePosition() {
            if (!this.e || !this.$refs.extraNetworks) return

            // getBoundingClientRect() returns viewport coordinates.  The popup
            // is fixed-positioned as well, so both now use the same coordinate
            // system (the old absolute + viewport-coordinate mix caused the
            // large vertical offset after scrolling).
            const anchorRect = this.e.getBoundingClientRect()
            const popupRect = this.$refs.extraNetworks.getBoundingClientRect()
            const gap = 4
            const margin = 10

            let left = anchorRect.left
            let top = anchorRect.bottom + gap

            // Keep the popup inside the viewport horizontally.
            if (left + popupRect.width > window.innerWidth - margin) {
                left = window.innerWidth - popupRect.width - margin
            }
            left = Math.max(margin, left)

            // If there is not enough room below, prefer the space above the tag.
            if (top + popupRect.height > window.innerHeight - margin) {
                const topAbove = anchorRect.top - popupRect.height - gap
                if (topAbove >= margin) {
                    top = topAbove
                } else {
                    top = Math.max(margin, window.innerHeight - popupRect.height - margin)
                }
            }
            top = Math.max(margin, top)

            this.style.top = Math.round(top) + 'px'
            this.style.left = Math.round(left) + 'px'
        },
        onPreviewLoad() {
            this.$nextTick(() => this.updatePosition())
        },
        _hide() {
            if (!this.isShow) return
            if (this.hideTimer) clearTimeout(this.hideTimer)
            this.hideTimer = setTimeout(() => {
                this.hideTimer = null
                if (!this.mouseIn && !this.eMouseIn) {
                    this.isShow = false
                }
            }, 10)
        },
        hide() {
            this.eMouseIn = false
            this._hide()
        },
        onMouseEnter() {
            this.mouseIn = true
        },
        onMouseMove() {
            this.mouseIn = true
        },
        onMouseLeave() {
            this.mouseIn = false
            this._hide()
        },
        copy(text) {
            this.$copyText(text).then(() => {
                this.$toastr.success(this.getLang('success'))
            }).catch(() => {
                this.$toastr.error(this.getLang('failed'))
            })
        },
        getCivitaiUrl(modelId) {
            return globals.civitaiUrl + '/models/' + modelId
        },
        onOpenCivitaiClick() {
            if (!this.data.modelId) return
            let url = this.getCivitaiUrl(this.data.modelId)
            window.open(url)
        },
        /*onUseKeywordsClick() {
            if (!this.data.trainedWords || !this.data.trainedWords.length) return
            if (!this.useCallback) return
            if (typeof this.useCallback !== 'function') return
            this.useCallback(this.data.trainedWords)
            this.$toastr.success(this.getLang('success'))
        },*/
        onUseKeywordsClick(value) {
            if (!value) return
            if (typeof this.useCallback !== 'function') return
            let tags = common.splitTags(value)
            console.log(tags)
            this.useCallback(tags)
            // this.$toastr.success(this.getLang('success'))
        },
    },
}
</script>