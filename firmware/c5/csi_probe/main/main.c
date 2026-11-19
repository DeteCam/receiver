/*
 * csi_probe: ESP32-C5 하나를 '보내는 쪽' 또는 '받는 쪽'으로 쓰는 빠른 시험용 펌웨어.
 *
 *  보내는 쪽(tx): 정해진 채널에서 ESP-NOW 프레임을 초당 hz 개 보낸다 (HT20, MCS0).
 *  받는 쪽(rx):   그 프레임의 CSI 를 받아 USB 로 한 줄씩 내보낸다.
 *
 * USB 로 한 줄 명령을 받는다. 설정은 NVS 에 남아, 다시 켜면 같은 설정으로 바로 시작한다
 * (보내는 쪽은 노트북 없이 보조배터리로 켜 두면 된다).
 *
 *   role tx | role rx       역할
 *   ch 6 | ch 36 | ch 149   채널 (1~13 은 2.4 GHz, 36 이상은 5 GHz). 대역폭은 HT20 고정
 *   hz 100                  보내는 빠르기 (tx)
 *   gain auto | gain lock   rx: 수신 이득을 그대로 둘지, 다음 100개의 가운데 값으로 묶을지 (바로 적용)
 *   status                  지금 설정
 *   reboot                  다시 켜기
 *
 * 받는 쪽이 내보내는 줄 (쉼표로 나눔):
 *   C,<seq>,<rssi>,<noise>,<agc>,<fft>,<comp*1000>,<ch>,<fmt>,<len>,<first_word_invalid>,<CSI 바이트 16진수>
 *   CSI 는 서브캐리어마다 2바이트 (허수, 실수 순서, 부호 있는 8비트). 해석은 bench/probe.py 가 한다.
 * 그 밖의 줄은 'I,' 로 시작한다.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_now.h"
#include "esp_system.h"
#if __has_include("esp_csi_gain_ctrl.h")
#include "esp_csi_gain_ctrl.h"
#define HAVE_GAIN_CTRL 1
#else
#define HAVE_GAIN_CTRL 0          /* 부품 저장소에서 esp_csi_gain_ctrl 을 받지 못했을 때: 이득 값 없이 CSI 만 낸다 */
#endif

#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#else
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#endif

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define IF_STA ESP_MAC_WIFI_STA
#else
#define IF_STA ESP_IF_WIFI_STA
#endif

static const uint8_t TX_MAC[6] = {0x1a, 0x00, 0x00, 0x00, 0x0c, 0x05};   /* 보내는 쪽의 MAC (받는 쪽이 이것만 고른다) */
static const uint8_t BCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

typedef struct {
    char role[4];      /* "tx" / "rx" */
    int32_t ch;
    int32_t hz;
    int32_t lock;      /* 1 = 수신 이득을 묶는다 */
} cfg_t;

static cfg_t s_cfg = {"rx", 6, 100, 0};

/* ---------- 설정 저장 ---------- */
static void cfg_load(void)
{
    nvs_handle_t h;
    if (nvs_open("probe", NVS_READONLY, &h) != ESP_OK) return;
    size_t n = sizeof(s_cfg.role);
    nvs_get_str(h, "role", s_cfg.role, &n);
    nvs_get_i32(h, "ch", &s_cfg.ch);
    nvs_get_i32(h, "hz", &s_cfg.hz);
    nvs_get_i32(h, "lock", &s_cfg.lock);
    nvs_close(h);
}

static void cfg_save(void)
{
    nvs_handle_t h;
    if (nvs_open("probe", NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_str(h, "role", s_cfg.role);
    nvs_set_i32(h, "ch", s_cfg.ch);
    nvs_set_i32(h, "hz", s_cfg.hz);
    nvs_set_i32(h, "lock", s_cfg.lock);
    nvs_commit(h);
    nvs_close(h);
}

static bool is_5g(int ch) { return ch > 14; }

/* ---------- Wi-Fi ---------- */
static void radio_set_channel(int ch)
{
    ESP_ERROR_CHECK(esp_wifi_set_band_mode(is_5g(ch) ? WIFI_BAND_MODE_5G_ONLY : WIFI_BAND_MODE_2G_ONLY));
    esp_err_t e = esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE);
    if (e != ESP_OK) printf("I,error,set_channel %d: %s\n", ch, esp_err_to_name(e));
}

static void radio_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_start());
    wifi_protocols_t protocols = {.ghz_2g = WIFI_PROTOCOL_11N, .ghz_5g = WIFI_PROTOCOL_11N};
    ESP_ERROR_CHECK(esp_wifi_set_protocols(IF_STA, &protocols));
    wifi_bandwidths_t bw = {.ghz_2g = WIFI_BW_HT20, .ghz_5g = WIFI_BW_HT20};
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(IF_STA, &bw));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    radio_set_channel(s_cfg.ch);
    if (strcmp(s_cfg.role, "tx") == 0) ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, TX_MAC));
}

static void now_init(void)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_peer_info_t peer = {.channel = 0, .ifidx = WIFI_IF_STA, .encrypt = false};
    memcpy(peer.peer_addr, BCAST, 6);
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    esp_now_rate_config_t rate = {.phymode = WIFI_PHY_MODE_HT20, .rate = WIFI_PHY_RATE_MCS0_LGI, .ersu = false, .dcm = false};
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(BCAST, &rate));
}

/* ---------- 보내는 쪽 ---------- */
static void tx_task(void *arg)
{
    uint32_t seq = 0, err = 0;
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        int hz = s_cfg.hz > 0 ? s_cfg.hz : 100;
        if (esp_now_send(BCAST, (const uint8_t *)&seq, sizeof(seq)) != ESP_OK) err++;
        seq++;
        if (seq % (hz * 5) == 0) printf("I,tx,ch=%ld,sent=%lu,err=%lu\n", (long)s_cfg.ch, (unsigned long)seq, (unsigned long)err);
        TickType_t period = pdMS_TO_TICKS(1000 / hz);
        vTaskDelayUntil(&last, period ? period : 1);
    }
}

/* ---------- 받는 쪽 ---------- */
#define CSI_MAX 512
typedef struct {
    uint32_t seq;
    int8_t rssi, noise;
    uint8_t agc;
    int8_t fft;
    int16_t comp_milli;
    uint8_t ch, fmt, fwi;
    uint16_t len;
    int8_t buf[CSI_MAX];
} csi_rec_t;

static QueueHandle_t s_q;
static volatile uint32_t s_drop = 0;
static volatile bool s_regain = false;   /* gain 명령: 이득 기준을 처음부터 다시 잡는다 */

static void csi_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || memcmp(info->mac, TX_MAC, 6)) return;
    static int n = 0;
    static uint8_t agc_base = 0;
    static int8_t fft_base = 0;
    const wifi_pkt_rx_ctrl_t *rc = &info->rx_ctrl;
    uint8_t agc = 0;
    int8_t fft = 0;
    float comp = 1.0f;
#if HAVE_GAIN_CTRL
    if (s_regain) {
        s_regain = false;
        n = 0;
        esp_csi_gain_ctrl_set_rx_force_gain(0, 0);          /* 0, 0 = 묶음을 풀고 자동으로 */
        esp_csi_gain_ctrl_reset_rx_gain_baseline();
    }
    esp_csi_gain_ctrl_get_rx_gain(rc, &agc, &fft);
    if (n < 100) {
        esp_csi_gain_ctrl_record_rx_gain(agc, fft);
    } else if (n == 100) {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_base, &fft_base);
        if (s_cfg.lock) esp_csi_gain_ctrl_set_rx_force_gain(agc_base, fft_base);
        printf("I,gain_baseline,agc=%d,fft=%d,lock=%ld\n", agc_base, fft_base, (long)s_cfg.lock);
    }
    esp_csi_gain_ctrl_get_gain_compensation(&comp, agc, fft);
#else
    (void)agc_base; (void)fft_base;
#endif
    n++;

    csi_rec_t r;
    r.seq = (info->payload && info->payload_len >= 19) ? *(uint32_t *)(info->payload + 15) : 0;
    r.rssi = rc->rssi;
    r.noise = rc->noise_floor;
    r.agc = agc;
    r.fft = fft;
    r.comp_milli = (int16_t)(comp * 1000.0f);
    r.ch = rc->channel;
    r.fmt = rc->cur_bb_format;
    r.fwi = info->first_word_invalid;
    r.len = info->len > CSI_MAX ? CSI_MAX : info->len;
    memcpy(r.buf, info->buf, r.len);
    if (xQueueSend(s_q, &r, 0) != pdTRUE) s_drop++;
}

static void print_task(void *arg)
{
    static csi_rec_t r;
    static char line[64 + CSI_MAX * 2];
    static const char hx[] = "0123456789abcdef";
    uint32_t got = 0;
    for (;;) {
        if (xQueueReceive(s_q, &r, portMAX_DELAY) != pdTRUE) continue;
        int k = snprintf(line, 64, "C,%lu,%d,%d,%u,%d,%d,%u,%u,%u,%u,", (unsigned long)r.seq, r.rssi, r.noise,
                         r.agc, r.fft, r.comp_milli, r.ch, r.fmt, r.len, r.fwi);
        for (int i = 0; i < r.len; i++) {
            uint8_t b = (uint8_t)r.buf[i];
            line[k++] = hx[b >> 4];
            line[k++] = hx[b & 15];
        }
        line[k++] = '\n';
        line[k] = 0;
        fputs(line, stdout);
        if (++got % 500 == 0) printf("I,rx,got=%lu,drop=%lu\n", (unsigned long)got, (unsigned long)s_drop);
    }
}

static void rx_start(void)
{
    s_q = xQueueCreate(64, sizeof(csi_rec_t));
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    wifi_csi_config_t c = {
        .enable = true,
        .acquire_csi_legacy = false,
        .acquire_csi_force_lltf = false,
        .acquire_csi_ht20 = true,
        .acquire_csi_ht40 = false,
        .acquire_csi_vht = false,
        .acquire_csi_su = false,
        .acquire_csi_mu = false,
        .acquire_csi_dcm = false,
        .acquire_csi_beamformed = false,
        .acquire_csi_he_stbc_mode = 2,
        .val_scale_cfg = 0,
        .dump_ack_en = false,
        .reserved = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&c));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    xTaskCreate(print_task, "print", 4096, NULL, 4, NULL);
}

/* ---------- 명령 ---------- */
static void print_status(void)
{
    printf("I,status,role=%s,ch=%ld,band=%s,hz=%ld,gain=%s\n", s_cfg.role, (long)s_cfg.ch,
           is_5g(s_cfg.ch) ? "5G" : "2.4G", (long)s_cfg.hz, s_cfg.lock ? "lock" : "auto");
}

static void console_init(void)
{
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
    usb_serial_jtag_driver_config_t c = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    c.rx_buffer_size = 1024;
    c.tx_buffer_size = 4096;
    usb_serial_jtag_driver_install(&c);
    usb_serial_jtag_vfs_use_driver();
#else
    uart_driver_install(CONFIG_ESP_CONSOLE_UART_NUM, 1024, 4096, 0, NULL, 0);
    uart_vfs_dev_use_driver(CONFIG_ESP_CONSOLE_UART_NUM);
#endif
    setvbuf(stdin, NULL, _IONBF, 0);
}

static void cmd_task(void *arg)
{
    char buf[64];
    int n = 0;
    for (;;) {
        int c = fgetc(stdin);
        if (c == EOF) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
        if (c != '\n' && c != '\r') { if (n < (int)sizeof(buf) - 1) buf[n++] = (char)c; continue; }
        buf[n] = 0;
        n = 0;
        if (!buf[0]) continue;
        char *arg1 = strchr(buf, ' ');
        if (arg1) { *arg1++ = 0; while (*arg1 == ' ') arg1++; }
        bool need_reboot = false;
        if (!strcmp(buf, "role") && arg1 && (!strcmp(arg1, "tx") || !strcmp(arg1, "rx"))) {
            strncpy(s_cfg.role, arg1, sizeof(s_cfg.role) - 1);
            need_reboot = true;
        } else if (!strcmp(buf, "ch") && arg1) {
            int ch = atoi(arg1);
            if ((ch >= 1 && ch <= 13) || (ch >= 36 && ch <= 177)) {
                s_cfg.ch = ch;
                radio_set_channel(ch);
                s_regain = true;                            /* 대역이 바뀌면 이득 기준도 새로 */
            } else {
                printf("I,error,bad channel\n");
            }
        } else if (!strcmp(buf, "hz") && arg1) {
            int hz = atoi(arg1);
            if (hz >= 1 && hz <= 1000) s_cfg.hz = hz;
        } else if (!strcmp(buf, "gain") && arg1) {
            s_cfg.lock = !strcmp(arg1, "lock");
            s_regain = true;
        } else if (!strcmp(buf, "reboot")) {
            esp_restart();
        } else if (strcmp(buf, "status")) {
            printf("I,error,unknown command: %s\n", buf);
            continue;
        }
        cfg_save();
        print_status();
        if (need_reboot) {
            printf("I,reboot\n");
            fflush(stdout);
            vTaskDelay(pdMS_TO_TICKS(100));
            esp_restart();
        }
    }
}

void app_main(void)
{
    esp_err_t r = nvs_flash_init();
    if (r == ESP_ERR_NVS_NO_FREE_PAGES || r == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        r = nvs_flash_init();
    }
    ESP_ERROR_CHECK(r);
    console_init();
    cfg_load();
    radio_init();
    now_init();
    print_status();
    if (strcmp(s_cfg.role, "tx") == 0) {
        xTaskCreate(tx_task, "tx", 4096, NULL, 5, NULL);
    } else {
        rx_start();
    }
    xTaskCreate(cmd_task, "cmd", 4096, NULL, 3, NULL);
}
