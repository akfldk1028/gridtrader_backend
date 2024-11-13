from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
from datetime import datetime


def capture_binance_futures_chart(save_dir='captured_charts'):
    """
    바이낸스 선물 차트를 캡쳐하는 함수 (1분봉 + MACD, 볼린저밴드)
    Args:
        save_dir (str): 캡처된 이미지를 저장할 디렉토리 경로
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920x1080')
    chrome_options.add_argument('--log-level=3')  # 불필요한 로그 숨기기

    driver = None
    try:
        print("Chrome 드라이버 초기화 중...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        wait = WebDriverWait(driver, 20)

        print("바이낸스 선물 차트 페이지 로딩 중...")
        driver.get("https://www.binance.com/en/futures/BTCUSDT")
        time.sleep(5)  # 초기 로딩 대기

        # 쿠키 동의 처리
        try:
            cookie_button = wait.until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            cookie_button.click()
            time.sleep(2)
        except:
            print("쿠키 동의 버튼 없음")

        # 차트 로딩 대기
        print("차트 로딩 대기 중...")
        wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "trading-chart"))
        )
        time.sleep(5)

        # 타임프레임 변경
        print("타임프레임 설정 중...")
        try:
            # 현재 선택된 타임프레임 버튼 찾기
            current_timeframe = wait.until(
                EC.element_to_be_clickable((By.XPATH,
                    "//div[contains(@class, 'typography-caption0')]//div[contains(@class, 'whitespace-nowrap')]"
                ))
            )
            driver.execute_script("arguments[0].click();", current_timeframe)
            time.sleep(2)

            # Available 섹션에서 1m 옵션 찾기
            print("1분 옵션 찾는 중...")
            one_min_option = wait.until(
                EC.element_to_be_clickable((By.XPATH,
                    "//div[contains(@class, 'typography-caption1') and text()='1m']"
                ))
            )
            driver.execute_script("arguments[0].click();", one_min_option)
            time.sleep(3)

            print("타임프레임 설정 완료")

        except Exception as e:
            print(f"타임프레임 설정 중 오류: {e}")
            with open('timeframe_error.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)

        # 인디케이터 추가
        print("인디케이터 추가 중...")
        try:
            # 인디케이터 버튼 찾기
            indicator_button = wait.until(
                EC.element_to_be_clickable((By.XPATH,
                    "//div[@class='flex items-center cursor-pointer relative text-[--color-IconNormal] hover:text-[--color-SecondaryText]']"
                ))
            )
            driver.execute_script("arguments[0].click();", indicator_button)
            time.sleep(2)

            # MACD 추가
            print("MACD 추가 중...")
            macd_item = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'MACD')]"))
            )
            driver.execute_script("arguments[0].click();", macd_item)
            time.sleep(2)

            # 인디케이터 메뉴 다시 열기
            driver.execute_script("arguments[0].click();", indicator_button)
            time.sleep(2)

            # Bollinger Bands 추가
            print("볼린저 밴드 추가 중...")
            bb_item = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Bollinger')]"))
            )
            driver.execute_script("arguments[0].click();", bb_item)
            time.sleep(2)

        except Exception as e:
            print(f"인디케이터 추가 중 오류: {e}")
            with open('indicator_error.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)

        # 차트 렌더링 대기
        print("차트 렌더링 대기 중...")
        time.sleep(10)

        # 스크린샷 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(save_dir, f'binance_futures_chart_{timestamp}.png')

        print("스크린샷 저장 중...")
        driver.save_screenshot(screenshot_path)
        print(f"차트가 성공적으로 저장되었습니다: {screenshot_path}")

        return screenshot_path

    except Exception as e:
        print(f"에러 발생: {str(e)}")
        if driver:
            try:
                error_screenshot = os.path.join(save_dir, 'error_screenshot.png')
                driver.save_screenshot(error_screenshot)
                print(f"에러 상황 스크린샷 저장됨: {error_screenshot}")
                # 페이지 소스도 저장 (디버깅용)
                with open('error_page.html', 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)
            except:
                pass
        return None

    finally:
        if driver:
            driver.quit()
            print("브라우저 세션 종료")


if __name__ == "__main__":
    print("바이낸스 선물 차트 캡쳐 테스트 시작")
    result = capture_binance_futures_chart()
    if result:
        print(f"테스트 완료. 저장된 파일: {result}")
    else:
        print("테스트 실패")