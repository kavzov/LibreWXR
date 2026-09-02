# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
# Modifications Copyright (C) 2026 Igor Kavzov

import pytest

from librewxr.config import settings
from librewxr.main import app, project_metadata


pytestmark = pytest.mark.api


def test_openapi_offers_corresponding_source():
    schema = app.openapi()

    assert schema["info"]["license"]["name"] == "AGPL-3.0-or-later"
    assert schema["info"]["license"]["url"] == (
        "https://www.gnu.org/licenses/agpl-3.0.html"
    )
    assert schema["externalDocs"]["url"] == settings.source_url


async def test_api_root_offers_corresponding_source():
    response = await project_metadata()

    assert response["license"] == "AGPL-3.0-or-later"
    assert response["source"] == settings.source_url
    assert response["upstream"] == "https://github.com/JoshuaKimsey/LibreWXR"
