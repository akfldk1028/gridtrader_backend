from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import os
from datetime import datetime


def capture_upbit_chart(save_dir='captured_charts'):
    """
    업비트 차트를 캡쳐하는 함수 (1초 단위, 볼린저 밴드 포함)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920x1080')

    driver = None
    try:
        print("Chrome 드라이버 초기화 중...")
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 20)

        print("업비트 차트 페이지 로딩 중...")
        driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-DOGE")

        # 차트 로딩 대기
        print("차트 로딩 대기 중...")
        chart_element = wait.until(
            EC.presence_of_element_located((By.ID, "fullChartiq"))
        )

        # 기본 대기 시간
        time.sleep(3)

        print("1초 단위 차트로 변경 중...")
        # 시간 설정 메뉴 클릭
        time_menu = wait.until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "cq-menu.ciq-period")
            )
        )
        time_menu.click()
        time.sleep(1)

        # 1초 옵션 선택
        one_hour_option = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//cq-item[@stxtap=\"Layout.setPeriodicity(1,1,'minute')\"]")))
        one_hour_option.click()
        time.sleep(2)

        try:
            # MACD 추가
            print("STO 추가를 위한 인디케이터 메뉴 열기...")
            study_menu = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "cq-menu.ciq-studies")
                )
            )
            study_menu.click()
            time.sleep(1)

            print("STO 추가 중...")
            macd_item = wait.until(
                EC.presence_of_element_located((By.XPATH, "//cq-item[.//translate[@original='Stochastic Momentum Index']]"))
            )
            ActionChains(driver).move_to_element(macd_item).click().perform()
            time.sleep(1)



            # 볼린저 밴드 추가를 위해 메뉴 다시 열기
            print("볼린저 밴드 추가를 위한 인디케이터 메뉴 다시 열기...")
            study_menu.click()
            time.sleep(1)

            # 볼린저 밴드 추가
            print("볼린저 밴드 추가 중...")
            bb_item = wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, "//cq-item[.//translate[@original='Bollinger Bands']]")
                )
            )
            ActionChains(driver).move_to_element(bb_item).click().perform()
            time.sleep(1)


        except Exception as e:
            print(f"인디케이터 추가 중 오류: {str(e)}")
            if driver:
                print("현재 표시된 인디케이터 메뉴 항목들:")
                try:
                    indicators = driver.find_elements(By.CSS_SELECTOR, "cq-study-dialog cq-scroll cq-item")
                    for ind in indicators:
                        print(f"- {ind.get_attribute('outerHTML')}")
                except:
                    pass

        # 차트 렌더링 대기
        print("차트 렌더링 대기 중...")
        time.sleep(2)

        # 스크린샷 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(save_dir, f'btc_chart_{timestamp}.png')

        print("스크린샷 저장 중...")
        driver.save_screenshot(screenshot_path)
        print(f"차트가 성공적으로 저장되었습니다: {screenshot_path}")

        return screenshot_path

    except Exception as e:
        print(f"에러 발생: {str(e)}")
        if driver:
            # 에러 발생 시 페이지 소스 저장 (디버깅용)
            with open('error_page.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            # 에러 상황에서도 스크린샷 시도
            try:
                error_screenshot = os.path.join(save_dir, 'error_screenshot.png')
                driver.save_screenshot(error_screenshot)
                print(f"에러 상황 스크린샷 저장: {error_screenshot}")
            except:
                pass
        return None

    finally:
        if driver:
            driver.quit()
            print("브라우저 세션 종료")


if __name__ == "__main__":
    print("차트 캡쳐 테스트 시작")
    result = capture_upbit_chart()
    if result:
        print(f"테스트 완료. 저장된 파일: {result}")
    else:
        print("테스트 실패")