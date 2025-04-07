import requests
import pandas as pd

def fetch_data_to_excel(url, output_file="output.xlsx"):
    try:
        response = requests.get("https://backend-hum5.onrender.com/api/registrations")
        response.raise_for_status()  # Check for HTTP errors
        json_data = response.json()

        if json_data.get("success") and "data" in json_data:
            rows = []
            for item in json_data["data"]:
                row = {
                    "Name": item.get("name"),
                    "Email": item.get("email"),
                    "Branch": item.get("branch"),
                    "Section": item.get("section"),
                    "Year": item.get("year"),
                    "Roll No": item.get("rollNo"),
                    "College": item.get("college"),
                    "Total Amount": item.get("totalAmount"),
                    "Transaction ID": item.get("transactionId"),
                    "Payment Link": item.get("paymentScreenshotLink"),
                    "Registration Date": item.get("registrationDate")
                }

                # Flatten selected events
                if item.get("selectedEvents"):
                    titles = ", ".join(event.get("title", "") for event in item["selectedEvents"])
                    row["Selected Events"] = titles
                else:
                    row["Selected Events"] = ""

                rows.append(row)

            # Create DataFrame and export to Excel
            df = pd.DataFrame(rows)
            df.to_excel(output_file, index=False)
            print(f"✅ Excel file saved as {output_file}")
        else:
            print("❌ Invalid response structure or no data found.")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
fetch_data_to_excel("https://your-api-url.com/endpoint")
